import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
def test_smallx(self):
    epsilon = 0.1 ** np.arange(1, 14)
    x = np.array([0.571173265106, 0.441027698518, 0.374219690278, 0.331392659217, 0.300820537459, 0.277539353999, 0.259023494805, 0.243829561254, 0.231063086389, 0.220135543236, 0.210641372041, 0.202290283658, 0.19487060742])
    dataset = np.column_stack([x, 1 - epsilon])
    FuncData(kolmogorov, dataset, (0,), 1, rtol=_rtol).check()