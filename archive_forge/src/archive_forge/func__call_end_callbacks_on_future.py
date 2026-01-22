from collections import defaultdict
from typing import Any, Dict, List, Optional
from warnings import warn
import torch
import torch.cuda
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import _ExperimentalConfig
from torch.autograd import (
from torch.autograd.profiler_util import (
from torch.futures import Future
def _call_end_callbacks_on_future(self, fut: Future[Any]) -> Future[Any]:
    """Use for profiling async calls that return a future.

        Calling this function will extend recording beyond this scope, until the future is
        satisfied. It is useful for profiling the end to end time of asynchronous calls.
        This function should only be called once to attach the callback onto the future, and
        will throw if called multiple times.

        Args:
            fut: (torch._C.Future): future for which to schedule
            callback for.

        Returns:
            A future that completes with the value of the passed in future when
            the profiling callbacks have ran.

        """
    if not self.run_callbacks_on_exit:
        raise RuntimeError('_call_end_callbacks_on_future can only be called once.')
    self.run_callbacks_on_exit = False
    record = self.record
    assert record is not None
    if not torch.jit.is_scripting():
        with torch._C.DisableTorchFunctionSubclass():
            profiled_future = torch.ops.profiler._call_end_callbacks_on_jit_fut._RecordFunction(record, fut)
    else:
        profiled_future = torch.ops.profiler._call_end_callbacks_on_jit_fut(record, fut)
    return profiled_future