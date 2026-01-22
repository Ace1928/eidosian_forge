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
class Test_ROS_MaxCen_GT_MaxUncen(Test_ROS_HelselAppendixB):
    res = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 3.0, 7.0, 9.0, 12.0, 15.0, 20.0, 27.0, 33.0, 50.0, 60, 70])
    cen = numpy.array([True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, True, True])