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
class TensorPipeAgentRpcTest(RpcAgentTestFixture, RpcTestCommon):

    def test_mismatched_type_for_options(self):
        rpc_backend_options = FooBackendOptions(self.init_method)
        with self.assertRaisesRegex(TypeError, '`rpc_backend_options` must be a `TensorPipeRpcBackendOptions`'):
            rpc.init_rpc(name=worker_name(self.rank), rank=self.rank, world_size=self.world_size, backend=rpc.BackendType.TENSORPIPE, rpc_backend_options=rpc_backend_options)

    def test_infer_backend_from_options(self):
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(init_method=self.init_method, _transports=tp_transports())
        rpc.init_rpc(name=worker_name(self.rank), rank=self.rank, world_size=self.world_size, rpc_backend_options=rpc_backend_options)
        self.assertIsInstance(rpc.api._get_current_rpc_agent(), rpc.TensorPipeAgent)

    @dist_init(setup_rpc=False)
    def test_set_and_get_num_worker_threads(self):
        NUM_THREADS = 27
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(init_method=self.rpc_backend_options.init_method, num_worker_threads=NUM_THREADS, _transports=tp_transports())
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=rpc_backend_options)
        info = rpc.api._get_current_rpc_agent().get_debug_info()
        self.assertEqual(int(info['agent.thread_pool_size']), NUM_THREADS)
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    def test_tensorpipe_set_default_timeout(self):
        timeout = 100
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(init_method=self.rpc_backend_options.init_method, num_worker_threads=self.rpc_backend_options.num_worker_threads, rpc_timeout=timeout, _transports=tp_transports())
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=rpc_backend_options)
        default_timeout = rpc.get_rpc_timeout()
        self.assertEqual(default_timeout, timeout)
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    def test_tensorpipe_options_throw_on_timedelta_timeout(self):
        from datetime import timedelta
        timeout = timedelta()
        with self.assertRaisesRegex(TypeError, 'incompatible constructor arguments'):
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(init_method=self.rpc_backend_options.init_method, num_worker_threads=self.rpc_backend_options.num_worker_threads, rpc_timeout=timeout)

    @dist_init
    def _test_rref_get_type_timeout(self, blocking):
        dst_rank = (self.rank + 1) % self.world_size
        dst = worker_name(dst_rank)
        slow_rref = rpc.remote(dst, MyClass, args=(torch.ones(2, 2), True))
        timeout = 0.5
        expected_err = self.get_timeout_error_regex()
        if blocking:
            with self.assertRaisesRegex(RuntimeError, expected_err):
                slow_rref._get_type(timeout=timeout, blocking=blocking)
        else:
            fut = slow_rref._get_type(timeout=timeout, blocking=blocking)
            with self.assertRaisesRegex(RuntimeError, expected_err):
                fut.wait()
        slow_rref.to_here()

    def test_rref_get_type_timeout_blocking(self):
        self._test_rref_get_type_timeout(blocking=True)

    def test_rref_get_type_timeout_non_blocking(self):
        self._test_rref_get_type_timeout(blocking=False)

    @dist_init
    def test_op_with_invalid_args(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(RuntimeError, 'Overloaded torch operator invoked from Python failed to many any schema'):
            rpc.rpc_sync(dst, torch.add, args=())

    def _test_rref_proxy_timeout(self, rref_proxy_api):
        dst_rank = (self.rank + 1) % self.world_size
        dst = worker_name(dst_rank)
        rref = rpc.remote(dst, MyClass, args=(torch.ones(2, 2),))
        rref.to_here()
        rref_api = getattr(rref, rref_proxy_api)
        self.assertTrue(rref_api is not None, f'Failed to get RRef proxy api: {rref_proxy_api}')
        expected_error = self.get_timeout_error_regex()
        timeout = 2
        with self.assertRaisesRegex(RuntimeError, expected_error):
            result = rref_api(timeout=timeout).my_slow_method(torch.ones(2, 2))
            if rref_api == rref.rpc_async:
                result.wait()
            elif rref_api == rref.remote:
                result._get_future().wait()
        slow_rref = rpc.remote(dst, MyClass, args=(torch.ones(2, 2), True))
        timeout = 0.01
        rref_api = getattr(slow_rref, rref_proxy_api)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            result = rref_api(timeout=timeout).my_instance_method(torch.ones(2, 2))
            if rref_api == slow_rref.rpc_async:
                result.wait()
        slow_rref.to_here()

    @dist_init
    def test_rref_proxy_timeout(self):
        for rpc_api in ['rpc_sync', 'rpc_async', 'remote']:
            self._test_rref_proxy_timeout(rpc_api)

    @dist_init
    def test_send_to_rank_sparse(self):
        dst_rank = (self.rank + 1) % self.world_size
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            x = build_sparse_tensor()
            y = build_sparse_tensor()
            expected_tensor = x + y
            ret = self._run_func_in_mode(dst_rank, torch.add, exec_mode, args=(x, y))
            self.assertEqual(expected_tensor, ret)
        for exec_mode in [RPCExecMode.SYNC, RPCExecMode.ASYNC, RPCExecMode.REMOTE]:
            x = build_sparse_tensor(coalesce=True)
            y = build_sparse_tensor(coalesce=True)
            expected_tensor = x + y
            ret = self._run_func_in_mode(dst_rank, torch.add, exec_mode, args=(x, y))
            self.assertEqual(expected_tensor, ret)

    @dist_init
    def test_self_py_udf_remote_sparse(self):
        self._self_py_udf_remote(rpc.get_worker_info(), build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor())

    @dist_init
    def test_self_remote_rref_as_rpc_arg_sparse(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        self._self_remote_rref_as_rpc_arg(dst, build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor())

    @dist_init
    def test_self_remote_rref_as_self_rpc_arg_sparse(self):
        self._self_remote_rref_as_rpc_arg(rpc.get_worker_info(), build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor())

    @dist_init
    def test_self_remote_rref_as_remote_arg_sparse(self):
        dst = worker_name((self.rank + 1) % self.world_size)
        self._self_remote_rref_as_remote_arg(dst, build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor())

    @dist_init
    def test_self_remote_rref_as_self_remote_arg_sparse(self):
        self._self_remote_rref_as_remote_arg(rpc.get_worker_info(), build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor())

    def test_world_size_one_sparse(self):
        self._world_size_one(build_sparse_tensor(), build_sparse_tensor())

    @dist_init
    def test_multi_rpc_sparse(self):
        self._multi_rpc(True)

    def test_wait_all_workers_sparse(self):
        self._wait_all_workers(heavy_rpc_sparse, build_sparse_tensor())

    def test_wait_all_workers_twice_sparse(self):
        self._wait_all_workers_twice(heavy_rpc_sparse, build_sparse_tensor())

    @dist_init
    def test_py_sparse_tensors_in_container(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        a = [build_sparse_tensor(), build_sparse_tensor()]
        ret = rpc.rpc_sync(worker_name(dst_rank), my_container_sum, args=(a,))
        self.assertEqual(ret, my_container_sum(a))

    @dist_init
    def test_nested_rpc_sparse(self):
        self._nested_rpc(nested_rpc_sparse, build_sparse_tensor() * 2)

    @dist_init
    def test_stress_heavy_rpc_sparse(self):
        self._stress_test_rpc(heavy_rpc_sparse, repeat=20, args=(build_sparse_tensor(),))

    @dist_init
    def test_builtin_remote_ret_sparse(self):
        self._builtin_remote_ret(build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor() * 2)

    @dist_init
    def test_builtin_remote_self_sparse(self):
        self._builtin_remote_self(build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor() * 2)

    @dist_init
    def test_multi_builtin_remote_ret_sparse(self):
        self._test_multi_remote_call(torch.add, True, args_fn=RpcTest._multi_args_fn)

    @dist_init
    def test_multi_py_udf_remote_sparse(self):
        self._test_multi_remote_call(my_function, True, kwargs_fn=RpcTest._multi_kwargs_fn)

    @dist_init
    def test_py_rref_args_sparse(self):
        self._py_rref_args(build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor() * 4)

    @dist_init
    def test_py_rref_args_user_share_sparse(self):
        self._py_rref_args_user_share(build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor() * 6)

    @dist_init
    def test_py_rpc_rref_args_sparse(self):
        self._py_rpc_rref_args(build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor(), build_sparse_tensor() * 6)

    @dist_init
    def test_nested_remote_sparse(self):
        self._nested_remote(nested_remote_sparse, build_sparse_tensor() + build_sparse_tensor())

    @dist_init
    def test_nested_rref_sparse(self):
        self._nested_rref(nested_rref_sparse, build_sparse_tensor() * 2, build_sparse_tensor() * 2)

    @dist_init
    def test_nested_rref_stress_sparse(self):
        self._nested_rref_stress(nested_rref_sparse, build_sparse_tensor() * 2, build_sparse_tensor() * 2)

    @dist_init
    def test_my_parameter_server_sparse(self):
        self._my_parameter_server(True)

    @dist_init(setup_rpc=False)
    def test_dynamic_rpc_init_rpc(self):
        rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, rpc_backend_options=self.rpc_backend_options)
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    def test_dynamic_rpc_new_rank_can_communicated_with_existing_rank(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        if self.rank == 0:
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, rpc_backend_options=self.rpc_backend_options)
        dist.barrier()
        if self.rank != 0:
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, rpc_backend_options=self.rpc_backend_options)
            result = rpc.rpc_sync(worker_name(0), torch.add, args=(torch.tensor(1), torch.tensor(1)))
            self.assertEqual(torch.add(torch.tensor(1), torch.tensor(1)), result)
        dist.barrier()
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    def test_dynamic_rpc_existing_rank_can_communicate_with_new_rank(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        if self.rank == 0:
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, rpc_backend_options=self.rpc_backend_options)
        dist.barrier()
        if self.rank != 0:
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, rpc_backend_options=self.rpc_backend_options)
        dist.barrier()
        if self.rank == 0:
            for i in range(1, self.world_size):
                result = rpc.rpc_sync(worker_name(i), torch.add, args=(torch.tensor(1), torch.tensor(1)))
                self.assertEqual(torch.add(torch.tensor(1), torch.tensor(1)), result)
        dist.barrier()
        rpc.shutdown()

    @skip_if_lt_x_gpu(2)
    @dist_init(setup_rpc=False)
    def test_dynamic_rpc_existing_rank_can_communicate_with_new_rank_cuda(self):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        if self.rank == 0:
            options = self.rpc_backend_options
            for i in range(1, self.world_size):
                dst = worker_name(i)
                options.set_device_map(dst, {1: 0})
                options.set_device_map(dst, {0: 1})
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, rpc_backend_options=options)
        dist.barrier()
        if self.rank != 0:
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, rpc_backend_options=self.rpc_backend_options)
        dist.barrier()
        rpc.shutdown()

    @dist_init(setup_rpc=False)
    def test_dynamic_rpc_init_rpc_without_rank(self):
        with self.assertRaisesRegex(ValueError, 'rank parameter missing'):
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rpc_backend_options=self.rpc_backend_options)
        with self.assertRaisesRegex(ValueError, 'environment variable RANK expected'):
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(init_method='env://')
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rpc_backend_options=rpc_backend_options)
        with self.assertRaisesRegex(ValueError, 'rank parameter missing'):
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(init_method='tcp://127.0.0.1:23456')
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rpc_backend_options=rpc_backend_options)

    @dist_init(setup_rpc=False)
    def test_dynamic_and_static_init_rpc_together(self):
        dist.init_process_group(backend='gloo', init_method=self.file_init_method, rank=self.rank, world_size=self.world_size)
        world_size_minus_one = self.world_size - 1
        if self.rank < world_size_minus_one:
            rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=world_size_minus_one, rpc_backend_options=self.rpc_backend_options)
        dist.barrier()
        if self.rank == world_size_minus_one:
            with self.assertRaisesRegex(RuntimeError, 'RPC group mixes statically and dynamically initialized members which is not supported.'):
                rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, rpc_backend_options=self.rpc_backend_options)