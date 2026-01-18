import itertools
import unittest
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..optpkg import optional_package
from ..casting import sctypes_aliases, shared_range, type_info
from ..spatialimages import HeaderDataError
from ..spm99analyze import HeaderTypeError, Spm99AnalyzeHeader, Spm99AnalyzeImage
from ..testing import (
from ..volumeutils import _dt_min_max, apply_read_scaling
from . import test_analyze
def test_write_scaling(self):
    for slope, inter, e_slope, e_inter in ((1, None, 1, None), (0, None, 1, None), (np.inf, None, 1, None), (2, None, 2, None)):
        self._check_write_scaling(slope, inter, e_slope, e_inter)