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
class Test_ROS_HelselAppendixB(CheckROSMixin):
    """
    Appendix B dataset from "Estimation of Descriptive Statists for
    Multiply Censored Water Quality Data", Water Resources Research,
    Vol 24, No 12, pp 1997 - 2004. December 1988.
    """
    decimal = 2
    res = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 3.0, 7.0, 9.0, 12.0, 15.0, 20.0, 27.0, 33.0, 50.0])
    cen = numpy.array([True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False])
    rescol = 'obs'
    cencol = 'cen'
    df = pandas.DataFrame({rescol: res, cencol: cen})
    expected_final = numpy.array([0.47, 0.85, 1.11, 1.27, 1.76, 2.34, 2.5, 3.0, 3.03, 4.8, 7.0, 9.0, 12.0, 15.0, 20.0, 27.0, 33.0, 50.0])
    expected_cohn = pandas.DataFrame({'nuncen_above': numpy.array([3.0, 6.0, numpy.nan]), 'nobs_below': numpy.array([6.0, 12.0, numpy.nan]), 'ncen_equal': numpy.array([6.0, 3.0, numpy.nan]), 'prob_exceedance': numpy.array([0.55556, 0.33333, 0.0])})