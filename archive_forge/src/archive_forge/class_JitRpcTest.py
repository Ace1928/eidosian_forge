import time
import io
from typing import Dict, List, Tuple, Any
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.distributed.rpc import RRef
from torch.distributed.rpc.internal import RPCExecMode, _build_rpc_profiling_key
from torch.futures import Future
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.autograd.profiler_legacy import profile as _profile
class JitRpcTest(RRefAPITest, RRefTypingTest, LocalRRefTest, JitRpcOpTest, FutureTypingTest, RpcAgentTestFixture):

    @dist_init
    def test_torchscript_function(self):
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        local_ret = one_arg(torch.ones(2, 2))
        ret = rpc.rpc_sync(dst_worker_name, one_arg, args=(torch.ones(2, 2),))
        self.assertEqual(ret, local_ret)
        rref = rpc.remote(dst_worker_name, one_arg, args=(torch.ones(2, 2),))
        self.assertEqual(rref.to_here(), local_ret)
        local_rref = rpc.remote(worker_name(self.rank), one_arg, args=(torch.ones(2, 2),))
        self.assertEqual(local_rref.to_here(), local_ret)

    @dist_init
    def test_torchscript_function_exception(self):
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(RuntimeError, 'one_arg\\(\\) expected at most'):
            ret = rpc.rpc_sync(dst_worker_name, one_arg, args=(10, 20))
        with self.assertRaisesRegex(RuntimeError, 'one_arg\\(\\) expected at most'):
            rref = rpc.remote(dst_worker_name, one_arg, args=(10, 20))

    @dist_init
    def test_torchscript_functions_not_supported(self):
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        my_local_script_module = MyScriptModule(self.rank)
        initialize_pg(self.file_init_method, self.rank, self.world_size)
        dist.barrier()
        ret = rpc.rpc_sync(dst_worker_name, MyScriptClass, args=(self.rank,))
        with self.assertRaisesRegex(TypeError, 'pickle'):
            ret = rpc.rpc_async(dst_worker_name, my_local_script_module.forward, args=())

    @dist_init
    def test_remote_script_module(self):
        import torch.distributed.rpc.api as api
        api._ignore_rref_leak = True
        local_ret = torch.ones(self.rank) + torch.ones(self.rank)
        n = self.rank + 1
        dst_rank = n % self.world_size
        remote_ref = rpc.remote(worker_name(dst_rank), construct_my_script_module, args=(self.rank,))
        ret = rpc.rpc_sync(worker_name(dst_rank), run_ref_script_module, args=(remote_ref, torch.ones(self.rank)))
        self.assertEqual(ret, local_ret)
        with self.assertRaisesRegex(RuntimeError, "is an RRef to a ScriptModule. It can't be sent through RPC from owner,"):
            ret = rpc.rpc_sync(worker_name(self.rank), run_ref_script_module, args=(remote_ref, torch.ones(self.rank)))

    @dist_init
    def test_create_script_module_on_remote(self):
        dst_name = worker_name((self.rank + 1) % self.world_size)
        created_script_module = rpc.rpc_sync(dst_name, MyScriptModule, args=(self.rank,))
        self.assertTrue(isinstance(created_script_module, torch.jit.ScriptModule))
        rank_ones_tensor = created_script_module()
        self.assertEqual(torch.ones(self.rank), rank_ones_tensor)
        remote_script_module = rpc.remote(dst_name, MyScriptModule, args=(self.rank,))
        remote_end_is_script = rpc.rpc_sync(remote_script_module.owner(), rref_isinstance, args=(remote_script_module, torch.jit.ScriptModule))
        self.assertTrue(remote_end_is_script)
        remote_forward_output = remote_script_module.rpc_sync().forward()
        self.assertEqual(remote_forward_output, torch.ones(self.rank))
        remote_func_output = remote_script_module.rpc_sync().custom_func()
        self.assertEqual(remote_func_output, torch.ones(self.rank))
        local_script_module = remote_script_module.to_here()
        self.assertTrue(isinstance(local_script_module, torch.jit.ScriptModule))
        rank_ones_tensor = local_script_module()
        self.assertEqual(rank_ones_tensor, torch.ones(self.rank))
        local_script_func_output = local_script_module.custom_func()
        self.assertEqual(local_script_func_output, torch.ones(self.rank))

    @dist_init
    def test_load_script_module_with_pickled_rref(self):
        dst_name = worker_name((self.rank + 1) % self.world_size)
        m1 = MyScriptModuleWithRRefs(dst_name)
        m2 = MyScriptModuleWithRRefs(dst_name)
        f = io.BytesIO()
        rpc._enable_jit_rref_pickle()
        torch.jit.save(m1, f)
        rpc._disable_jit_rref_pickle()
        out1 = rpc.rpc_sync(dst_name, load_script_module_with_pickled_rref, args=(f.getvalue(),))
        out2 = m2()
        self.assertEqual(out1, out2)

    @dist_init
    def test_rref_jit_pickle_not_supported(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_var = rpc_return_rref(worker_name(dst_rank))
        with TemporaryFileName() as fname:
            with self.assertRaisesRegex(RuntimeError, 'RRef jit pickling is only allowed inside RPC calls'):
                save_rref(rref_var, fname)

    @dist_init
    def test_remote_script_throw(self):
        rref = rpc.remote(worker_name((self.rank + 1) % self.world_size), script_raise_func, args=(torch.ones(2),))
        with self.assertRaisesRegex(Exception, '.*Expected error.*'):
            rref.to_here()

    @dist_init
    def test_remote_script_udf(self):
        rref = rpc.remote(worker_name((self.rank + 1) % self.world_size), script_fork_wait_udf, args=(torch.ones(2),))
        self.assertEqual(rref.to_here(), torch.ones(2) * 2)

    @dist_init
    def test_async_script_udf(self):
        future = rpc.rpc_async(worker_name((self.rank + 1) % self.world_size), script_fork_wait_udf, args=(torch.ones(2),))
        self.assertEqual(future.wait(), torch.ones(2) * 2)

    @dist_init
    def test_callback_simple(self):

        def callback(fut):
            return fut.wait() + 1
        future = rpc.rpc_async(worker_name((self.rank + 1) % self.world_size), script_fork_wait_udf, args=(torch.ones(2),)).then(callback)
        self.assertEqual(future.wait(), torch.ones(2) * 2 + 1)

    @dist_init
    def test_callback_chain(self):
        n = self.rank + 1
        dst = worker_name(n % self.world_size)

        def callback(fut):
            return fut.wait() + 1
        fut = rpc.rpc_async(worker_name(n % self.world_size), one_arg, args=(torch.ones(n, n),))
        num_cbs = 20
        for _ in range(num_cbs):
            fut = fut.then(callback)
        self.assertEqual(fut.wait(), torch.ones(n, n) + 1 + num_cbs)

    @dist_init
    def test_add_done_callback(self):
        callback_called = None

        def callback(fut):
            nonlocal callback_called
            callback_called = fut.wait() * 2
        future = rpc.rpc_async(worker_name((self.rank + 1) % self.world_size), script_fork_wait_udf, args=(torch.ones(2),))
        future.add_done_callback(callback)
        future_then = future.then(lambda _: True)
        self.assertEqual(future.wait(), torch.ones(2) * 2)
        future_then.wait()
        self.assertEqual(callback_called, torch.ones(2) * 4)

    @dist_init
    def test_async_script_throw(self):
        future = rpc.rpc_async(worker_name((self.rank + 1) % self.world_size), script_fork_wait_throw, args=(torch.ones(2),))
        with self.assertRaisesRegex(Exception, '.*Expected error.*'):
            future.wait()

    @dist_init
    def test_callback_with_exception(self):

        def callback(fut):
            with self.assertRaisesRegex(Exception, '.*Expected error.*'):
                fut.wait()
            raise RuntimeError('Another expected error')
        future = rpc.rpc_async(worker_name((self.rank + 1) % self.world_size), script_fork_wait_throw, args=(torch.ones(2),)).then(callback)
        with self.assertRaisesRegex(RuntimeError, 'Another expected error'):
            future.wait()

    @dist_init
    def test_call_rpc_with_profiling(self):
        if self.rank == 0:
            with _profile() as prof:
                prof_key = _build_rpc_profiling_key(RPCExecMode.ASYNC, torch._jit_internal._qualified_name(one_arg), 'worker0', 'worker1')
                with torch.autograd.profiler.record_function(prof_key) as rf:
                    ret = call_rpc_with_profiling(rf.record, 'worker1')
            events = prof.function_events
            function_event = get_function_event(events, prof_key)
            self.assertTrue(torch._jit_internal._qualified_name(one_arg) in function_event.name)

    @dist_init
    def test_rpc_async_jit_profiled(self):
        if self.rank == 0:
            dst_rank = (self.rank + 1) % self.world_size
            dst_worker_name = worker_name(dst_rank)
            args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
            kwargs = {}
            with _profile() as prof:
                script_rpc_async_call(dst_worker_name, args, kwargs)
            function_events = prof.function_events
            qual_name = torch._jit_internal._qualified_name(two_args_two_kwargs)
            rpc_async_jit_event = [event for event in function_events if qual_name in event.name and event.node_id == self.rank]
            self.assertEqual(len(rpc_async_jit_event), 1)
            rpc_async_jit_event = rpc_async_jit_event[0]
            profiled_name = _build_rpc_profiling_key(RPCExecMode.ASYNC_JIT, qual_name, worker_name(self.rank), dst_worker_name)
            self.assertEqual(profiled_name, rpc_async_jit_event.name)
            remote_events = [event for event in function_events if event.is_remote]
            remote_event_node_ids = {remote_event.node_id for remote_event in remote_events}
            self.assertEqual(remote_event_node_ids, {dst_rank})
            remote_add = next((remote_event for remote_event in remote_events if 'aten::add' in remote_event.name))
            remote_add_profiled_name = f'{profiled_name}#remote_op: aten::add'
            self.assertEqual(remote_add.name, remote_add_profiled_name)

    @dist_init
    def test_record_function_on_caller_rpc_async(self):
        if self.rank == 0:
            dst_rank = (self.rank + 1) % self.world_size
            dst_worker_name = worker_name(dst_rank)
            block_scope = 'foo'
            with _profile() as prof:
                record_function_on_caller_rpc_async(dst_worker_name, block_scope)
            function_events = prof.function_events
            record_function_scope_event = [event for event in function_events if event.name == block_scope]
            self.assertEqual(1, len(record_function_scope_event))
            record_function_scope_event = record_function_scope_event[0]
            expected_key = _build_rpc_profiling_key(RPCExecMode.ASYNC_JIT, torch._jit_internal._qualified_name(script_add_ones), worker_name(self.rank), dst_worker_name)
            jit_rpc_events = [event for event in function_events if event.name == expected_key]
            self.assertEqual(2, len(jit_rpc_events))
            for jit_rpc_event in jit_rpc_events:
                self.assertTrue(record_function_scope_event.cpu_time_total > jit_rpc_event.cpu_time_total)

    @dist_init
    def test_rpc_torchscript_record_function(self):
        REMOTE_OP_STR = '#remote_op: '
        if self.rank == 0:
            dst_rank = (self.rank + 1) % self.world_size
            dst_worker_name = worker_name(dst_rank)
            block_scope = 'foo'
            with _profile() as prof:
                call_rpc_torchscript_with_record_function(dst_worker_name, block_scope)
            prof.key_averages()
            function_events = prof.function_events
            expected_key = _build_rpc_profiling_key(RPCExecMode.ASYNC_JIT, torch._jit_internal._qualified_name(script_add_ones_with_record_function), worker_name(self.rank), dst_worker_name) + REMOTE_OP_STR + block_scope
            remote_record_function_event = next((evt for evt in function_events if evt.name == expected_key))
            self.assertTrue(block_scope in remote_record_function_event.name)
            remote_children = remote_record_function_event.cpu_children
            self.assertTrue(('aten::add' in child.name for child in remote_children))

    def test_record_function_jit_end_callbacks_with_fork(self):
        sleep_interval = 1
        with _profile() as prof:
            with torch.autograd.profiler.record_function('foo') as rf:
                fut = torch.jit._fork(sleep, sleep_interval)
                rf._call_end_callbacks_on_future(fut)
            fut.wait()
        function_events = prof.function_events
        sleep_event = get_function_event(function_events, 'foo')
        self.assertEqual(sleep_event.name, 'foo')
        self.assertGreaterAlmostEqual(sleep_event.cpu_time * 1e-06, sleep_interval)

    def test_call_fork_in_jit_with_profiling(self):
        with _profile() as prof:
            with torch.autograd.profiler.record_function('foo') as rf:
                ret = call_fork_with_profiling(rf.record)
        events = prof.function_events
        function_event = get_function_event(events, 'foo')
        self.assertEqual(function_event.name, 'foo')

    @dist_init
    def test_async_function_simple(self):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        ret = rpc.rpc_sync(dst1, async_add, args=(dst2, torch.ones(2, 2), torch.ones(2, 2)))
        self.assertEqual(ret, torch.ones(2, 2) + 1)

    @dist_init
    def test_async_function_wrong_return_type(self):
        with self.assertRaisesRegex(RuntimeError, 'Async functions must return an IValue of Future type, but got Tensor'):
            rpc.rpc_sync(worker_name((self.rank + 1) % self.world_size), async_wrong_type)

    @dist_init
    def test_async_function_wrong_decorator_order(self):
        with self.assertRaises(RuntimeError):

            @torch.jit.script
            @rpc.functions.async_execution
            def async_wrong_decorator_order(to: str, x: Tensor, y: Tensor) -> Future[Tensor]:
                return rpc.rpc_async(to, script_add, (x, y))

    @dist_init
    def test_async_function_remote(self):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        rref = rpc.remote(dst1, async_add, args=(dst2, torch.ones(2, 2), torch.ones(2, 2)))
        self.assertEqual(rref.to_here(), torch.ones(2, 2) + 1)

    @dist_init
    def test_async_function_remote_multi(self):
        dst1 = worker_name((self.rank + 1) % self.world_size)
        dst2 = worker_name((self.rank + 2) % self.world_size)
        num = 20
        rrefs = []
        for i in range(num):
            rrefs.append(rpc.remote(dst1, async_add, args=(dst2, torch.ones(2, 2), torch.ones(2, 2) * i)))
        for i in range(num):
            self.assertEqual(rrefs[i].to_here(), torch.ones(2, 2) + i)

    @dist_init
    def test_async_function_wrong_return_type_remote(self):
        rref = rpc.remote(worker_name((self.rank + 1) % self.world_size), async_wrong_type)
        with self.assertRaisesRegex(RuntimeError, 'Async functions must return an IValue of Future type, but got Tensor'):
            rref.to_here()