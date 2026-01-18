import math as _math
import warnings as _warnings
import numpy as _numpy
import cupy as _cupy
from cupyx.profiler._time import _repeat, _PerfCaseResult  # for tests  # NOQA
 Timing utility for measuring time spent by both CPU and GPU.

    This function is a very convenient helper for setting up a timing test. The
    GPU time is properly recorded by synchronizing internal streams. As a
    result, to time a multi-GPU function all participating devices must be
    passed as the ``devices`` argument so that this helper knows which devices
    to record. A simple example is given as follows:

    .. code-block:: py

        import cupy as cp
        from cupyx.time import repeat

        def f(a, b):
            return 3 * cp.sin(-a) * b

        a = 0.5 - cp.random.random((100,))
        b = cp.random.random((100,))
        print(repeat(f, (a, b), n_repeat=1000))


    Args:
        func (callable): a callable object to be timed.
        args (tuple): positional argumens to be passed to the callable.
        kwargs (dict): keyword arguments to be passed to the callable.
        n_repeat (int): number of times the callable is called. Increasing
            this value would improve the collected statistics at the cost
            of longer test time.
        name (str): the function name to be reported. If not given, the
            callable's ``__name__`` attribute is used.
        n_warmup (int): number of times the callable is called. The warm-up
            runs are not timed.
        max_duration (float): the maximum time (in seconds) that the entire
            test can use. If the taken time is longer than this limit, the test
            is stopped and the statistics collected up to the breakpoint is
            reported.
        devices (tuple): a tuple of device IDs (int) that will be timed during
            the timing test. If not given, the current device is used.

    Returns:
        :class:`~cupyx.profiler._time._PerfCaseResult`:
            an object collecting all test results.

    .. warning::
        This API is moved to :func:`cupyx.profiler.benchmark` since CuPy v10.
        Access through ``cupyx.time`` is deprecated.
    