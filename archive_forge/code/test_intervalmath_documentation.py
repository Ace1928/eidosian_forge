from sympy.plotting.intervalmath import interval
from sympy.testing.pytest import raises

    test that interval objects are hashable.
    this is required in order to be able to put them into the cache, which
    appears to be necessary for plotting in py3k. For details, see:

    https://github.com/sympy/sympy/pull/2101
    https://github.com/sympy/sympy/issues/6533
    