import concurrent.futures
import contextlib
import json
import os
import sys
import threading
import time
from collections import namedtuple
from functools import partial
from threading import Event
from threading import Lock
from unittest import mock
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.rpc import RRef, _get_debug_info, _rref_context_get_debug_info, WorkerInfo
from torch.distributed.rpc.api import _use_rpc_pickler, _thread_local_var, _wait_all
from torch.distributed.rpc.internal import (
from torch.futures import Future
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_utils import TemporaryFileName
from torch.autograd.profiler_legacy import profile as _profile
class RpcTestCommon:

    def _run_func_in_mode(self, to, fn, mode, args=None, kwargs=None):
        if mode == RPCExecMode.SYNC:
            return rpc.rpc_sync(to, fn, args=args, kwargs=kwargs)
        elif mode == RPCExecMode.ASYNC:
            return rpc.rpc_async(to, fn, args=args, kwargs=kwargs).wait()
        elif mode == RPCExecMode.REMOTE:
            return rpc.remote(to, fn, args=args, kwargs=kwargs).to_here()

    def _self_py_udf_remote(self, worker_info, x, y, z):
        rref = rpc.remote(worker_info, my_function, args=(x, y, z))
        self.assertEqual(rref.to_here(), x + y + z)

    def _self_remote_rref_as_rpc_arg(self, dst, x, y, z):
        self_worker_info = rpc.get_worker_info()
        rref = rpc.remote(self_worker_info, my_function, args=(x, y, z))
        fut = rpc.rpc_async(dst, add_rref_to_value, args=(rref, x))
        ret = rpc.rpc_sync(dst, add_rref_to_value, args=(rref, x + y))
        self.assertEqual(ret, x + y + z + x + y)
        self.assertEqual(fut.wait(), x + y + z + x)

    def _self_remote_rref_as_remote_arg(self, dst, x, y, z):
        self_worker_info = rpc.get_worker_info()
        rref = rpc.remote(self_worker_info, my_function, args=(x, y, z))
        ret_rref = rpc.remote(dst, add_rref_to_value, args=(rref, x))
        self.assertEqual(ret_rref.to_here(), x + y + z + x)

    def _world_size_one(self, a, b):
        if self.rank == 0:
            rpc.init_rpc(name='me', backend=self.rpc_backend, rank=0, world_size=1, rpc_backend_options=self.rpc_backend_options)

            def _rpc_sync(x, y):
                expect = x * 2
                result = rpc.rpc_sync('me', my_tensor_function, args=(x, y))
                self.assertEqual(expect, result)

            def _rpc_async(x, y):
                expect = x * 2
                result = rpc.rpc_async('me', my_tensor_function, args=(x, y)).wait()
                self.assertEqual(expect, result)

            def _remote(x, y):
                expect = x * 2
                result = rpc.remote('me', my_tensor_function, args=(x, y)).to_here()
                self.assertEqual(expect, result)
            _rpc_sync(a, b)
            _rpc_async(a, b)
            _remote(a, b)
            rpc.shutdown()

    def _multi_rpc(self, sparse):
        dst_rank = (self.rank + 1) % self.world_size
        for i in range(20):
            n = i + self.rank + 1
            if sparse:
                x = build_sparse_tensor() * n
                y = build_sparse_tensor() * n
            else:
                x = torch.ones(2, 2)
                y = torch.ones(2, 2)
            ret = rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(x, y))
            self.assertEqual(ret, x * 2)

    def _run_uneven_workload(self, f, x, num_repeat=30):
        if self.rank == 0:
            self.assertTrue(self.world_size >= 3)
            dst = 'worker1'
            futs = []
            for _ in range(num_repeat):
                fut = rpc.rpc_async(dst, f, args=(x,))
                futs.append(fut)
            for fut in torch.futures.collect_all(futs).wait():
                self.assertEqual(fut.wait(), 0)
            dst = 'worker2'
            futs = []
            for _ in range(num_repeat):
                fut = rpc.rpc_async(dst, f, args=(x,))
                futs.append(fut)
            for val in torch.futures.wait_all(futs):
                self.assertEqual(val, 0)

    def _wait_all_workers(self, f, x):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        rpc.init_rpc(name='worker%d' % self.rank, backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        self._run_uneven_workload(f, x)
        rpc.api._wait_all_workers()
        dist.barrier()
        rpc.shutdown(graceful=False)

    def _wait_all_workers_twice(self, f, x):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        rpc.init_rpc(name='worker%d' % self.rank, backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
        self._run_uneven_workload(f, x)
        rpc.api._wait_all_workers()
        rpc.api._wait_all_workers()
        dist.barrier()
        rpc.shutdown(graceful=False)

    def _nested_rpc(self, f, expected):
        n = self.rank + 1
        dst_rank = n % self.world_size
        ret = rpc.rpc_sync(worker_name(dst_rank), f, args=(worker_name(self.rank),))
        self.assertEqual(ret, expected)

    def _stress_test_rpc(self, f, repeat=1000, args=()):
        n = self.rank + 1
        dst_rank = n % self.world_size
        futs = []
        tik = time.time()
        for _ in range(repeat):
            fut = rpc.rpc_async(worker_name(dst_rank), f, args=args)
            futs.append(fut)
        for val in torch.futures.wait_all(futs):
            self.assertEqual(val, 0)
        tok = time.time()
        print(f'Rank {self.rank} finished testing {repeat} times in {tok - tik} seconds.')

    def _builtin_remote_ret(self, x, y, expected):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref = rpc.remote(worker_name(dst_rank), torch.add, args=(x, y))
        self.assertEqual(rref.to_here(), expected)

    def _builtin_remote_self(self, x, y, expected):
        rref = rpc.remote(worker_name(self.rank), torch.add, args=(x, y))
        self.assertEqual(rref.local_value(), expected)

    def _test_multi_remote_call(self, fn, sparse, args_fn=lambda x, y: (), kwargs_fn=lambda x, y: {}):
        m = 10
        n = self.rank + 1
        dst_rank = n % self.world_size
        rrefs = []
        expected = []
        for i in range(m):
            n = n + i
            rrefs.append(rpc.remote(worker_name(dst_rank), fn, args=args_fn(n, sparse), kwargs=kwargs_fn(n, sparse)))
            expected.append(fn(*args_fn(n, sparse), **kwargs_fn(n, sparse)))
        for i in range(m):
            self.assertEqual(rrefs[i].to_here(), expected[i])

    def _py_rref_args(self, a, b, x, y, expected):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_a = rpc.remote(worker_name(dst_rank), torch.add, args=(a, b))
        rref_b = rpc.remote(worker_name(dst_rank), torch.add, args=(x, y))
        rref_c = rpc.remote(worker_name(dst_rank), my_rref_function, args=(rref_a, rref_b))
        self.assertEqual(rref_c.to_here(), expected)

    def _py_rref_args_user_share(self, a, b, c, x, y, z, expected):
        n = self.rank + 1
        owner_rank = n % self.world_size
        user_rank = (n + 1) % self.world_size
        rref_a = rpc.remote(worker_name(owner_rank), my_function, args=(a, b, c))
        rref_b = rpc.remote(worker_name(owner_rank), my_function, args=(x, y, z))
        rref_c = rpc.remote(worker_name(user_rank), my_rref_function, args=(rref_a, rref_b))
        self.assertEqual(rref_c.to_here(), expected)

    def _py_rpc_rref_args(self, a, b, c, x, y, z, expected):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_a = rpc.remote(worker_name(dst_rank), my_function, args=(a, b, c))
        rref_b = rpc.remote(worker_name(dst_rank), my_function, args=(x, y, z))
        c = rpc.rpc_sync(worker_name(dst_rank), my_rref_function, args=(rref_a, rref_b))
        self.assertEqual(c, expected)

    def _nested_remote(self, f, expected):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        rref = rpc.remote(worker_name(dst_rank1), f, args=(worker_name(dst_rank2),))
        self.assertEqual(rref.to_here(), expected)

    def _nested_rref(self, f, expected1, expected2):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        rref_of_rrefs = rpc.remote(worker_name(dst_rank1), f, args=(worker_name(dst_rank2),))
        rrefs = rref_of_rrefs.to_here()
        self.assertEqual(len(rrefs), 2)
        self.assertEqual(rrefs[0].to_here(), expected1)
        self.assertEqual(rrefs[1].to_here(), expected2)

    def _nested_rref_stress(self, f, expected1, expected2):
        n = self.rank + 1
        dst_rank1 = n % self.world_size
        dst_rank2 = (n + 1) % self.world_size
        all_rrefs = []
        for _ in range(20):
            all_rrefs.append(rpc.remote(worker_name(dst_rank1), f, args=(worker_name(dst_rank2),)))
        for i in range(20):
            rref_of_rrefs = all_rrefs[i]
            rrefs = rref_of_rrefs.to_here()
            self.assertEqual(len(rrefs), 2)
            self.assertEqual(rrefs[0].to_here(), expected1)
            self.assertEqual(rrefs[1].to_here(), expected2)

    def _trainer_func(self, rref, sparse):
        m = MyEmbeddingBagModel(sparse=sparse)
        loss_fn = nn.MSELoss()
        for i in range(10):
            outputs = m(torch.rand(10, 10).long())
            loss_fn(outputs, torch.rand(10, 10)).backward()
            gradient = next(iter(m.parameters())).grad
            fut = rref.rpc_async().average(rref, i, gradient)
            gradient = fut.wait()
            if gradient.is_sparse:
                gradient = gradient.to_dense().double()
            ps_gradient = rref.rpc_sync().get_gradient(rref)
            if ps_gradient.is_sparse:
                ps_gradient = ps_gradient.to_dense().double()
            self.assertTrue(torch.equal(gradient, ps_gradient))

    def _my_parameter_server(self, sparse):
        ps_rref = RRef(MyParameterServer(self.world_size - 1))
        futures = []
        for index in range(1, self.world_size):
            futures.append(rpc.rpc_async(worker_name((self.rank + index) % self.world_size), self._trainer_func, args=(ps_rref, sparse)))
        torch.futures.wait_all(futures)

    def _test_cuda_future_extraction(self, wrapper, unwrapper, sparse_tensor):
        future = Future(devices=['cuda:0'])
        with torch.cuda.device('cuda:0'):
            stream = torch.cuda.Stream()
            another_stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                if sparse_tensor:
                    tensor = build_sparse_tensor().to('cuda:0')
                    add_tensor = build_sparse_tensor().to('cuda:0')
                    expected_tensor = (tensor + add_tensor).coalesce()
                else:
                    tensor = torch.zeros((100,), device='cuda:0')
                    add_tensor = torch.ones((100,), device='cuda:0')
                    expected_tensor = tensor + add_tensor
                torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
                tensor += add_tensor
                if sparse_tensor:
                    tensor = tensor.coalesce()
                future.set_result(wrapper(tensor))
            with torch.cuda.stream(another_stream):
                tensor = unwrapper(future.wait())
                if sparse_tensor:
                    self.assertTrue(torch.eq(tensor.indices(), expected_tensor.indices()).all().item())
                    self.assertTrue(torch.eq(tensor.values(), expected_tensor.values()).all().item())
                    self.assertEqual(tensor.size(), expected_tensor.size())
                else:
                    self.assertTrue(torch.eq(tensor, expected_tensor).all().item())