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
class JitRpcOpTest:

    @dist_init
    def test_all_kwargs_are_populated_by_defaults(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {}
        for script_op in [script_rpc_async_call, script_rpc_sync_call, script_rpc_remote_call]:
            ret = script_op(dst_worker_name, args, kwargs)
            self.assertEqual(ret, torch.tensor([10, 10]))

    @dist_init
    def test_some_kwargs_are_populated_by_defaults(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {'first_kwarg': torch.tensor([2, 2])}
        for script_op in [script_rpc_async_call, script_rpc_sync_call, script_rpc_remote_call]:
            ret = script_op(dst_worker_name, args, kwargs)
            self.assertEqual(ret, torch.tensor([9, 9]))

    @dist_init
    def test_no_kwargs_are_populated_by_defaults(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {'first_kwarg': torch.tensor([2, 2]), 'second_kwarg': torch.tensor([3, 3])}
        for script_op in [script_rpc_async_call, script_rpc_sync_call, script_rpc_remote_call]:
            ret = script_op(dst_worker_name, args, kwargs)
            self.assertEqual(ret, torch.tensor([8, 8]))

    @dist_init
    def test_args_and_kwargs_contain_different_types(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        @torch.jit.script
        def script_rpc_async_call_with_assorted_types(dst_worker_name: str):
            args = (torch.tensor([1, 1]), 'str_arg', 1)
            kwargs: Dict[str, Any] = {'tensor_kwarg': torch.tensor([3, 3]), 'str_kwarg': '_str_kwarg', 'int_kwarg': 3}
            fut = rpc.rpc_async(dst_worker_name, assorted_types_args_kwargs, args, kwargs)
            ret = fut.wait()
            return ret
        ret = script_rpc_async_call_with_assorted_types(dst_worker_name)
        self.assertEqual(ret, (torch.tensor([4, 4]), 'str_arg_str_kwarg', 4))

    @dist_init
    def test_kwargs_not_passed(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        @torch.jit.script
        def script_rpc_async_call_without_kwargs_passed(dst_worker_name: str):
            args = ()
            fut = rpc.rpc_async(dst_worker_name, no_arg, args)
            ret = fut.wait()
            return ret
        ret = script_rpc_async_call_without_kwargs_passed(dst_worker_name)
        self.assertEqual(ret, 0)

    @dist_init
    def test_args_kwargs_are_neither_passed(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        @torch.jit.script
        def script_rpc_async_call_without_args_kwargs_passed(dst_worker_name: str):
            fut = rpc.rpc_async(dst_worker_name, no_arg)
            ret = fut.wait()
            return ret
        ret = script_rpc_async_call_without_args_kwargs_passed(dst_worker_name)
        self.assertEqual(ret, 0)

    @dist_init
    def test_less_than_needed_args_are_specified(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(RuntimeError, 'Argument second_arg not provided'):

            @torch.jit.script
            def script_rpc_async_call_with_less_args(dst_worker_name: str):
                args = (torch.tensor([1, 1]),)
                kwargs = {}
                fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
                ret = fut.wait()
                return ret

    @dist_init
    def test_more_than_needed_args_are_specified(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(RuntimeError, 'Expected at most 4 arguments but found 5 positional arguments'):

            @torch.jit.script
            def script_rpc_async_call_with_more_args(dst_worker_name: str):
                args = (torch.tensor([1, 1]), torch.tensor([2, 2]), torch.tensor([3, 3]), torch.tensor([4, 4]), torch.tensor([5, 5]))
                kwargs = {}
                fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
                ret = fut.wait()
                return ret

    @dist_init
    def test_unexepected_kwarg_is_specified(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        @torch.jit.script
        def script_rpc_async_call_with_unexpected_kwarg(dst_worker_name: str):
            args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
            kwargs = {'third_kwarg': torch.tensor([1, 1])}
            fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
            ret = fut.wait()
            return ret
        with self.assertRaisesRegex(RuntimeError, "Unknown keyword argument 'third_kwarg'"):
            ret = script_rpc_async_call_with_unexpected_kwarg(dst_worker_name)
            self.assertEqual(ret, 0)

    @dist_init
    def test_call_python_function_remotely_from_script_not_supported(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        @torch.jit.script
        def rpc_async_call_remote_py_function_in_torchscript(dst_worker_name: str):
            args = ()
            kwargs = {}
            fut = rpc.rpc_async(dst_worker_name, python_function, args, kwargs)
            ret = fut.wait()
            return ret
        with self.assertRaisesRegex(RuntimeError, 'attempted to get undefined function'):
            ret = rpc_async_call_remote_py_function_in_torchscript(dst_worker_name)
            self.assertEqual(ret, 0)

    @dist_init
    def test_call_script_function_that_raises_remotely_from_script(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        @torch.jit.script
        def rpc_async_call_remote_raising_torchscript_in_torchscript(dst_worker_name: str):
            args = ()
            kwargs = {}
            fut = rpc.rpc_async(dst_worker_name, raise_script, args, kwargs)
            ret = fut.wait()
            return ret
        with self.assertRaisesRegex(RuntimeError, 'Expected error'):
            ret = rpc_async_call_remote_raising_torchscript_in_torchscript(dst_worker_name)
            self.assertEqual(ret, 0)

    @dist_init
    def test_call_script_function_that_not_exists_remotely_from_script(self):
        if self.rank != 0:
            return
        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        @torch.jit.script
        def nonexisting_script():
            return 0

        @torch.jit.script
        def rpc_async_call_remote_nonexisting_torchscript_in_torchscript(dst_worker_name: str):
            args = ()
            kwargs = {}
            fut = rpc.rpc_async(dst_worker_name, nonexisting_script, args, kwargs)
            ret = fut.wait()
            return ret
        with self.assertRaisesRegex(RuntimeError, 'attempted to get undefined function nonexisting_script'):
            ret = rpc_async_call_remote_nonexisting_torchscript_in_torchscript(dst_worker_name)
            self.assertEqual(ret, 0)