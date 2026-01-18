import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
def test_nonfinite():
    funcs = [('btdtria', 3), ('btdtrib', 3), ('bdtrik', 3), ('bdtrin', 3), ('chdtriv', 2), ('chndtr', 3), ('chndtrix', 3), ('chndtridf', 3), ('chndtrinc', 3), ('fdtridfd', 3), ('ncfdtr', 4), ('ncfdtri', 4), ('ncfdtridfn', 4), ('ncfdtridfd', 4), ('ncfdtrinc', 4), ('gdtrix', 3), ('gdtrib', 3), ('gdtria', 3), ('nbdtrik', 3), ('nbdtrin', 3), ('nrdtrimn', 3), ('nrdtrisd', 3), ('pdtrik', 2), ('stdtr', 2), ('stdtrit', 2), ('stdtridf', 2), ('nctdtr', 3), ('nctdtrit', 3), ('nctdtridf', 3), ('nctdtrinc', 3), ('tklmbda', 2)]
    np.random.seed(1)
    for func, numargs in funcs:
        func = getattr(sp, func)
        args_choices = [(float(x), np.nan, np.inf, -np.inf) for x in np.random.rand(numargs)]
        for args in itertools.product(*args_choices):
            res = func(*args)
            if any((np.isnan(x) for x in args)):
                assert_equal(res, np.nan)
            else:
                pass