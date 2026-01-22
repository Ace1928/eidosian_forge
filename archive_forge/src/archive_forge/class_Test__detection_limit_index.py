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
class Test__detection_limit_index:

    def setup_method(self):
        self.cohn = load_basic_cohn()
        self.empty_cohn = pandas.DataFrame(numpy.empty((0, 7)))

    def test_empty(self):
        assert_equal(ros._detection_limit_index(None, self.empty_cohn), 0)

    def test_populated(self):
        assert_equal(ros._detection_limit_index(3.5, self.cohn), 0)
        assert_equal(ros._detection_limit_index(6.0, self.cohn), 3)
        assert_equal(ros._detection_limit_index(12.0, self.cohn), 5)

    def test_out_of_bounds(self):
        with pytest.raises(IndexError):
            ros._detection_limit_index(0, self.cohn)