from statsmodels.compat.pandas import assert_series_equal, assert_frame_equal
from io import StringIO
from textwrap import dedent
import numpy as np
import numpy.testing as npt
import numpy
from numpy.testing import assert_equal
import pandas
import pytest
from statsmodels.imputation import ros
def test_populated(self):
    assert_equal(ros._detection_limit_index(3.5, self.cohn), 0)
    assert_equal(ros._detection_limit_index(6.0, self.cohn), 3)
    assert_equal(ros._detection_limit_index(12.0, self.cohn), 5)