import os
import sys
import time
from itertools import zip_longest
import numpy as np
from numpy.testing import assert_
import pytest
from scipy.special._testutils import assert_func_equal
def time_limited(timeout=0.5, return_val=np.nan, use_sigalrm=True):
    """
    Decorator for setting a timeout for pure-Python functions.

    If the function does not return within `timeout` seconds, the
    value `return_val` is returned instead.

    On POSIX this uses SIGALRM by default. On non-POSIX, settrace is
    used. Do not use this with threads: the SIGALRM implementation
    does probably not work well. The settrace implementation only
    traces the current thread.

    The settrace implementation slows down execution speed. Slowdown
    by a factor around 10 is probably typical.
    """
    if POSIX and use_sigalrm:

        def sigalrm_handler(signum, frame):
            raise TimeoutError()

        def deco(func):

            def wrap(*a, **kw):
                old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout)
                try:
                    return func(*a, **kw)
                except TimeoutError:
                    return return_val
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
            return wrap
    else:

        def deco(func):

            def wrap(*a, **kw):
                start_time = time.time()

                def trace(frame, event, arg):
                    if time.time() - start_time > timeout:
                        raise TimeoutError()
                    return trace
                sys.settrace(trace)
                try:
                    return func(*a, **kw)
                except TimeoutError:
                    sys.settrace(None)
                    return return_val
                finally:
                    sys.settrace(None)
            return wrap
    return deco