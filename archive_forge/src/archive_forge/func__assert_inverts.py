import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
def _assert_inverts(*a, **kw):
    d = _CDFData(*a, **kw)
    d.check()