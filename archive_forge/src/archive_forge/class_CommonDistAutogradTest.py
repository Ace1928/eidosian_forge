import sys
import threading
import time
from enum import Enum
import random
import torch
import torch.nn as nn
from datetime import timedelta
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.rpc import RRef
from torch.testing._internal.common_utils import IS_MACOS, skip_but_pass_in_sandcastle_if
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
class CommonDistAutogradTest(RpcAgentTestFixture):

    def _exec_func_with_dst(self, dst, exec_mode, method, *args):
        if ExecMode.LOCAL == exec_mode:
            if len(args) == 1 and isinstance(args[0], list):
                return method(*args[0])
            return method(*args)
        elif ExecMode.RPC_SYNC == exec_mode:
            return rpc.rpc_sync(worker_name(dst), method, args=args)
        elif ExecMode.REMOTE == exec_mode:
            return rpc.remote(worker_name(dst), method, args=args).to_here()
        elif ExecMode.RPC_ASYNC == exec_mode:
            fut = rpc.rpc_async(worker_name(dst), method, args=args)
            return fut.wait()
        else:
            raise ValueError(f'Unrecognized ExecMode {exec_mode}')

    def _exec_func(self, exec_mode, method, *args):
        return self._exec_func_with_dst(self._next_rank(), exec_mode, method, *args)

    def _next_rank(self):
        if hasattr(self, 'dst_rank'):
            self.dst_rank = (self.dst_rank + 1) % self.world_size
            if self.dst_rank == self.rank:
                return self._next_rank()
        else:
            self.dst_rank = (self.rank + 1) % self.world_size
        return self.dst_rank

    def _check_rpc_done(self, rank_distance):
        _check_rpc_done(rank_distance)

    def _verify_backwards(self, exec_mode, tensors, context_id, local_grads, *args):
        if exec_mode == ExecMode.LOCAL:
            torch.autograd.backward(tensors)
            return [arg.grad for arg in args]
        else:
            self._verify_backwards_remote(tensors, context_id, local_grads, *args)

    def _verify_backwards_remote(self, tensors, context_id, local_grads, *args):
        dist_autograd.backward(context_id, tensors)
        grads = dist_autograd.get_gradients(context_id)
        nargs = len(args)
        ngrads = 0
        for i in range(0, nargs):
            if local_grads[i] is not None:
                self.assertIn(args[i], grads)
                self.assertEqual(local_grads[i], grads[args[i]])
                ngrads += 1
            else:
                self.assertNotIn(args[i], grads)
        self.assertEqual(ngrads, len(grads))

    def _test_graph(self, fn, exec_mode, sparse):
        dst_rank = (self.rank + 1) % self.world_size
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        with dist_autograd.context() as context_id:
            if sparse:
                t1 = build_sparse_tensor()
                t2 = build_sparse_tensor()
            else:
                t1 = torch.ones(3, 3, requires_grad=True)
                t2 = torch.zeros(3, 3, requires_grad=True)
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), fn, args=(t1, t2))
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), fn, args=(t1, t2)).to_here()
            else:
                raise ValueError(f'Unrecognized ExecMode {exec_mode}')
            rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(1, len(recv_functions))
            self._verify_graph_for_first_rpc_call(next(iter(send_functions.values())), next(iter(recv_functions.values())), t1, t2, ret)
            self._check_rpc_done(1)
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            self._verify_graph_for_rpc_call_exec(next(iter(send_functions.values())))
            dist.barrier()
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._retrieve_context(context_id)
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._current_context()

    def _test_graph_for_py_nested_call(self, exec_mode, sparse):
        dst_rank = (self.rank + 1) % self.world_size
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        with dist_autograd.context() as context_id:
            if sparse:
                t1 = build_sparse_tensor(requires_grad=True)
                t2 = build_sparse_tensor(requires_grad=True)
            else:
                t1 = torch.ones(3, 3, requires_grad=True)
                t2 = torch.zeros(3, 3, requires_grad=True)
            nest_dst_rank = (dst_rank + 1) % self.world_size
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), my_py_nested_call, args=(t1, t2, dst_rank, self.world_size, 1))
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), my_py_nested_call, args=(t1, t2, dst_rank, self.world_size, 1)).to_here()
            else:
                raise ValueError(f'Unrecognized ExecMode {exec_mode}')
            dist.barrier()
            for rd in [1, 2, 3]:
                rpc.rpc_sync(worker_name((self.rank + rd) % self.world_size), _set_rpc_done, args=(context_id, rd))
            dist.barrier()
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(1, len(recv_functions))
            self._verify_graph_for_first_rpc_call(next(iter(send_functions.values())), next(iter(recv_functions.values())), t1, t2, ret)
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            self._verify_graph_for_nested_rpc_call(ctx)
            ctx = dist_autograd._retrieve_context(ctx_ids[2])
            self._verify_graph_for_nested_rpc_call(ctx)
            ctx = dist_autograd._retrieve_context(ctx_ids[3])
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            self._verify_graph_for_rpc_call_exec(next(iter(send_functions.values())))
            dist.barrier()

    def _test_graph_for_py_nested_call_itself(self, exec_mode, sparse):
        dst_rank = (self.rank + 1) % self.world_size
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        with dist_autograd.context() as context_id:
            if sparse:
                t1 = build_sparse_tensor(requires_grad=True)
                t2 = build_sparse_tensor(requires_grad=True)
            else:
                t1 = torch.ones(3, 3, requires_grad=True)
                t2 = torch.zeros(3, 3, requires_grad=True)
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), my_py_nested_call, args=(t1, t2, (self.rank - 1 + self.world_size) % self.world_size, self.world_size, 0))
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), my_py_nested_call, args=(t1, t2, (self.rank - 1 + self.world_size) % self.world_size, self.world_size, 0)).to_here()
            else:
                raise ValueError(f'Unrecognized ExecMode {exec_mode}')
            rpc.rpc_sync(worker_name((self.rank + 1) % self.world_size), _set_rpc_done, args=(context_id, 1))
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(2, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(2, len(recv_functions))
            self._verify_graph_for_first_rpc_call(next(iter(send_functions.values())), list(recv_functions.values())[1], t1, t2, ret)
            self._verify_graph_for_rpc_call_exec(list(send_functions.values())[1])
            self._check_rpc_done(1)
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            self._verify_graph_for_nested_rpc_call(ctx)
            dist.barrier()

    def _test_no_graph_with_tensors_not_require_grad(self, exec_mode, sparse):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dst_rank = (self.rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            if sparse:
                t1 = build_sparse_tensor(requires_grad=False)
                t2 = build_sparse_tensor(requires_grad=False)
            else:
                t1 = torch.ones(3, 3, requires_grad=False)
                t2 = torch.zeros(3, 3, requires_grad=False)
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), torch.add, args=(t1, t2))
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), torch.add, args=(t1, t2)).to_here()
            else:
                raise ValueError(f'Unrecognized ExecMode {exec_mode}')
            rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
            ctx = dist_autograd._current_context()
            send_functions = ctx._send_functions()
            self.assertEqual(len(send_functions), 0)
            recv_functions = ctx._recv_functions()
            self.assertEqual(len(recv_functions), 0)
            self._check_rpc_done(1)
            self.assertNotEqual(-1, dist_autograd._retrieve_context(ctx_ids[1]))
            dist.barrier()

    def _test_rpc_complex_args(self, exec_mode, sparse):
        with dist_autograd.context() as context_id:
            num_tensors = 10
            tensors = []
            for i in range(num_tensors):
                if sparse:
                    tensor = build_sparse_tensor(requires_grad=i % 2 == 0)
                else:
                    tensor = torch.ones(3, 3, requires_grad=i % 2 == 0)
                tensors.append(tensor)
            dst_rank = self._next_rank()
            if ExecMode.RPC_SYNC == exec_mode:
                ret = rpc.rpc_sync(worker_name(dst_rank), torch.stack, args=(tensors,))
            elif ExecMode.REMOTE == exec_mode:
                ret = rpc.remote(worker_name(dst_rank), torch.stack, args=(tensors,)).to_here()
            else:
                raise ValueError(f'Unrecognized ExecMode {exec_mode}')
            self.assertEqual(torch.stack(tensors), ret)
            next_funcs = next(iter(dist_autograd._current_context()._send_functions().values())).next_functions
            idx = 0
            for i in range(len(next_funcs)):
                self.assertEqual('torch::autograd::AccumulateGrad', next_funcs[i][0].name())
                self.assertEqual(tensors[i], next_funcs[i][0].variable)
            ctx = dist_autograd._current_context()
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(len(worker_ids), 1)
            self.assertEqual(worker_ids, {dst_rank})

    def context_cleanup_test_helper(self, rpc_args, func, nested=False):
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        if nested:
            dst_rank = (self.rank + 1) % self.world_size
            nested_dst_rank = (dst_rank + 1) % self.world_size
            dst_ranks = {dst_rank}
        else:
            dst_ranks = {rank for rank in range(self.world_size) if rank != self.rank}
        with dist_autograd.context() as context_id:
            for dst_rank in dst_ranks:
                rpc.rpc_sync(worker_name(dst_rank), func, args=rpc_args)
                rpc.rpc_sync(worker_name(dst_rank), _set_rpc_done, args=(context_id, 1))
                if nested:
                    rpc.rpc_sync(worker_name(nested_dst_rank), _set_rpc_done, args=(context_id, 2))
        with self.assertRaises(RuntimeError):
            dist_autograd._retrieve_context(context_id)
        dist.barrier()
        success = _all_contexts_cleaned_up()
        self.assertTrue(success)

    def _backward_no_grad_on_tensor(self, t1, t2, sparse):
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2))
            if sparse:
                loss = torch.sparse.sum(loss)
            else:
                loss = loss.sum()
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            self.assertIsNone(t1.grad)
            self.assertIsNone(t2.grad)
            loss_local = torch.add(t1, t2)
            if sparse:
                loss_local = torch.sparse.sum(loss_local)
            else:
                loss_local = loss_local.sum()
            loss_local.backward()
            self.assertIsNotNone(t1.grad)
            self.assertIsNotNone(t2.grad)
            t1_grad_before = t1.grad
            t2_grad_before = t2.grad
            dist_autograd.backward(context_id, [loss])
            self.assertEqual(t1_grad_before, t1.grad)
            self.assertEqual(t2_grad_before, t2.grad)

    def _backward_rref(self, callee, rref_owner, t1, t2, local_grads, sparse):
        local_ret = torch.add(t1, t2)
        if sparse:
            local_ret = torch.sparse.sum(local_ret)
        else:
            local_ret = local_ret.sum()
        local_ret.backward()
        with dist_autograd.context() as context_id:
            if sparse:
                rref_t1 = rpc.remote(rref_owner, build_sparse_tensor, args=(False, True))
            else:
                rref_t1 = rpc.remote(rref_owner, _torch_ones, args=((3, 3),), kwargs={'requires_grad': True})
            if callee == rref_owner:
                rref = rpc.remote(callee, my_rref_add, args=(rref_t1, t2))
            else:
                rref = rpc.remote(callee, my_nested_rref_add, args=(rref_owner, rref_t1, t2))
            ret = rref.to_here()
            if sparse:
                ret = torch.sparse.sum(ret)
            else:
                ret = ret.sum()
            dist_autograd.backward(context_id, [ret])
            grads = dist_autograd.get_gradients(context_id)
            self.assertIn(t2, grads)
            self.assertEqual(grads[t2], t2.grad)
            self.assertTrue(rpc.rpc_sync(rref_owner, _compare_owner_value, args=(context_id, rref_t1, t1.grad)))

    def _test_trainer_ps(self, create_ref_fn, trainer_fn, sparse):
        if sparse:
            t1 = build_sparse_tensor(requires_grad=True)
            t2 = build_sparse_tensor(requires_grad=True)
        else:
            t1 = torch.ones((3, 3), requires_grad=True)
            t2 = torch.zeros((3, 3), requires_grad=True)
        local_ret = torch.add(t1, t2)
        if sparse:
            torch.sparse.sum(local_ret).backward()
        else:
            local_ret.sum().backward()
        rref_t1 = rpc.remote(worker_name(self.rank), create_ref_fn, args=())
        rank_diffs = [1, 2, 3]
        futures = []
        for rank_diff in rank_diffs:
            futures.append(rpc.rpc_async(worker_name((self.rank + rank_diff) % self.world_size), trainer_fn, args=(rref_t1, t2, worker_name(self.rank), rank_diff, sparse)))
        for rank_diff in rank_diffs:
            self._check_rpc_done(rank_diff)
        accumulate_grad_func = None
        for rank_diff in rank_diffs:
            ctx_id = ctx_ids[rank_diff]
            grads = dist_autograd.get_gradients(ctx_id)
            local_t1 = rref_t1.to_here()
            self.assertIn(local_t1, grads)
            self.assertEqual(grads[local_t1], t1.grad)
        _set_rpc_done(None, 0)
        torch.futures.wait_all(futures)

    def _backward_multiple_round_trips(self, t1, t2, t3, t4, t5, local_grads, sparse):
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                val = self._exec_func(exec_mode, torch.add, t1, t2)
                val = self._exec_func(exec_mode, torch.mul, t3, val)
                s1 = self._exec_func(exec_mode, torch.stack, (t4, val))
                s2 = self._exec_func(exec_mode, torch.stack, (t5, val))
                if sparse:
                    val = self._exec_func(exec_mode, torch.mul, s1, s2)
                    val = self._exec_func(exec_mode, torch.mul, val, val)
                    loss = torch.sparse.sum(val)
                else:
                    val = self._exec_func(exec_mode, torch.bmm, s1, s2)
                    val = self._exec_func(exec_mode, torch.matmul, val, val)
                    loss = val.sum()
                ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2, t3, t4, t5)
                local_grads = ret if ret else local_grads

    def _backward_different_dtypes(self, t1, t2, sparse):
        local_grads = None
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                loss = self._exec_func(exec_mode, torch.add, t1, t2)
                if sparse:
                    loss = torch.sparse.sum(loss)
                else:
                    loss = loss.sum()
                local_grads = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)

    def _backward_simple_python_udf(self, t1, t2, sparse):
        local_grads = None
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func(exec_mode, my_py_add, t1, t2)
                if sparse:
                    loss = torch.sparse.sum(ret)
                else:
                    loss = ret.sum()
                local_grads = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)

    def _backward_simple_script_call(self, t1, t2, sparse):
        local_grads = None
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.RPC_ASYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                forward_ret = self._exec_func(exec_mode, my_script_add, t1, t2)
                if sparse:
                    loss = torch.sparse.sum(forward_ret)
                else:
                    loss = forward_ret.sum()
                ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)
                local_grads = ret if ret else local_grads

    def _nested_backward_accumulate_grads(self, t1, t2, sparse):
        with dist_autograd.context() as context_id:
            ret = rpc.rpc_sync(worker_name(self._next_rank()), DistAutogradTest._test_nested_backward_accumulate_grads, args=(t1, t2, self._next_rank()))
            if sparse:
                loss = torch.sparse.sum(ret)
            else:
                loss = ret.sum()
            dist_autograd.backward(context_id, [loss], retain_graph=True)
            dist_autograd.backward(context_id, [loss])

    def _backwards_nested_python_udf(self, t1, t2, sparse):
        t3 = t1 * t2
        t4 = t1 + t2
        res = t3 + t4
        loss = t1 * t2 * t3 * t4 * res
        if sparse:
            loss = torch.sparse.sum(loss)
        else:
            loss = loss.sum()
        torch.autograd.backward([loss])
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(worker_name(self._next_rank()), DistAutogradTest._nested_python_udf, args=(t1, t2, self._next_rank()))
            if sparse:
                loss = torch.sparse.sum(loss)
            else:
                loss = loss.sum()
            dist_autograd.backward(context_id, [loss])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(t1.grad, grads[t1])
            self.assertEqual(t2.grad, grads[t2])

    def _mixed_requires_grad(self, t1, t2, sparse):
        for exec_mode in [ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func(exec_mode, DistAutogradTest._mixed_requires_grad_operaton, t1, t2)
                self.assertEqual(t1 * t2, ret)
                if sparse:
                    loss = torch.sparse.sum(ret)
                else:
                    loss = ret.sum()
                dist_autograd.backward(context_id, [loss])
                self.assertTrue(t1.requires_grad)
                self.assertFalse(t2.requires_grad)
                grads = dist_autograd.get_gradients(context_id)
                self.assertIn(t1, grads)
                self.assertNotIn(t2, grads)
                self.assertEqual(t2, grads[t1])

    def _multiple_backward(self, t1, t2, sparse):
        with dist_autograd.context() as context_id:
            loss = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2))
            if sparse:
                loss = torch.sparse.sum(loss)
            else:
                loss = loss.sum()
            for i in range(1000):
                dist_autograd.backward(context_id, [loss], retain_graph=True)

    def _verify_graph_for_first_rpc_call(self, send_function, recv_function, t1, t2, ret):
        next_funcs = send_function.next_functions
        self.assertEqual(2, len(next_funcs))
        self.assertEqual('torch::autograd::AccumulateGrad', next_funcs[0][0].name())
        self.assertEqual(t1, next_funcs[0][0].variable)
        self.assertEqual(0, next_funcs[0][1])
        self.assertEqual('torch::autograd::AccumulateGrad', next_funcs[1][0].name())
        self.assertEqual(t2, next_funcs[1][0].variable)
        self.assertEqual(0, next_funcs[1][1])
        self.assertEqual(ret.grad_fn, recv_function)

    def _backward_simple(self, dst, t1, t2, local_grads, sparse):
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func_with_dst(dst, exec_mode, torch.add, t1, t2)
                if sparse:
                    loss = torch.sparse.sum(ret)
                else:
                    loss = ret.sum()
                ret = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)
                local_grads = ret if ret else local_grads

    def _verify_graph_for_rpc_call_exec(self, send_function):
        next_funcs = send_function.next_functions
        self.assertEqual(1, len(next_funcs))
        add_backward_fn = next_funcs[0][0]
        self.assertEqual('AddBackward0', add_backward_fn.name())
        next_funcs = add_backward_fn.next_functions
        self.assertEqual(2, len(next_funcs))
        self.assertEqual('torch::distributed::autograd::RecvRpcBackward', next_funcs[0][0].name())
        self.assertEqual('torch::distributed::autograd::RecvRpcBackward', next_funcs[1][0].name())
        self.assertEqual(next_funcs[0][0], next_funcs[1][0])

    def _verify_graph_for_nested_rpc_call(self, ctx):
        send_functions = ctx._send_functions()
        self.assertEqual(2, len(send_functions))
        next_funcs = next(iter(send_functions.values())).next_functions
        self.assertEqual(2, len(next_funcs))
        self.assertEqual('torch::distributed::autograd::RecvRpcBackward', next_funcs[0][0].name())
        self.assertEqual('torch::distributed::autograd::RecvRpcBackward', next_funcs[1][0].name())
        self.assertEqual(next_funcs[0][0], next_funcs[1][0])
        next_funcs = list(send_functions.values())[1].next_functions
        self.assertEqual(1, len(next_funcs))
        self.assertEqual('torch::distributed::autograd::RecvRpcBackward', next_funcs[0][0].name())