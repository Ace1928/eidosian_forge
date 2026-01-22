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
class emit_itt:
    """Context manager that makes every autograd operation emit an ITT range.

    It is useful when running the program under Intel(R) VTune Profiler::

        vtune <--vtune-flags> <regular command here>

    The Instrumentation and Tracing Technology (ITT) API enables your application to generate and
    control the collection of trace data during its execution across different Intel tools.
    This context manager is to annotate Intel(R) VTune Profiling trace. With help of this context manager,
    you will be able to see labled ranges in Intel(R) VTune Profiler GUI.

    .. warning:
        This context manager should not be called recursively, i.e. at most one
        instance should be enabled at any given time.

    Args:
        enabled (bool, optional): Setting ``enabled=False`` makes this context manager a no-op.
            Default: ``True``.
        record_shapes (bool, optional): If ``record_shapes=True``, the itt range wrapping
            each autograd op will append information about the sizes of Tensor arguments received
            by that op, in the following format:
            ``[[arg0.size(0), arg0.size(1), ...], [arg1.size(0), arg1.size(1), ...], ...]``
            Non-tensor arguments will be represented by ``[]``.
            Arguments will be listed in the order they are received by the backend op.
            Please note that this order may not match the order in which those arguments were passed
            on the Python side.  Also note that shape recording may increase the overhead of itt range creation.
            Default: ``False``

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> with torch.autograd.profiler.emit_itt():
        ...     model(x)

    """

    def __init__(self, enabled=True, record_shapes=False):
        self.enabled = enabled
        self.entered = False
        self.record_shapes = record_shapes

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError('ITT annotation context manager is not reentrant')
        self.entered = True
        _run_on_profiler_start()
        _enable_profiler(ProfilerConfig(ProfilerState.ITT, self.record_shapes, False, False, False, False, _ExperimentalConfig()), set())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        _disable_profiler()
        _run_on_profiler_stop()
        return False