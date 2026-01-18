import sys
import unittest
from numba.tests.support import captured_stdout
from numba.core.config import IS_WIN32
def timefunc(correct, s, func, *args, **kwargs):
    """
                Benchmark *func* and print out its runtime.
                """
    print(s.ljust(20), end=' ')
    res = func(*args, **kwargs)
    if correct is not None:
        assert np.allclose(res, correct), (res, correct)
    print('{:>5.0f} ms'.format(min(repeat(lambda: func(*args, **kwargs), number=5, repeat=2)) * 1000))
    return res