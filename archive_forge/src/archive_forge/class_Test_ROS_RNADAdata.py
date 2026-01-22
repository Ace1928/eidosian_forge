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
class Test_ROS_RNADAdata(CheckROSMixin):
    decimal = 3
    datastring = StringIO(dedent('        res cen\n        0.090  True\n        0.090  True\n        0.090  True\n        0.101 False\n        0.136 False\n        0.340 False\n        0.457 False\n        0.514 False\n        0.629 False\n        0.638 False\n        0.774 False\n        0.788 False\n        0.900  True\n        0.900  True\n        0.900  True\n        1.000  True\n        1.000  True\n        1.000  True\n        1.000  True\n        1.000  True\n        1.000 False\n        1.000  True\n        1.000  True\n        1.000  True\n        1.000  True\n        1.000  True\n        1.000  True\n        1.000  True\n        1.000  True\n        1.000  True\n        1.000  True\n        1.000  True\n        1.000  True\n        1.100 False\n        2.000 False\n        2.000 False\n        2.404 False\n        2.860 False\n        3.000 False\n        3.000 False\n        3.705 False\n        4.000 False\n        5.000 False\n        5.960 False\n        6.000 False\n        7.214 False\n       16.000 False\n       17.716 False\n       25.000 False\n       51.000 False'))
    rescol = 'res'
    cencol = 'cen'
    df = pandas.read_csv(datastring, sep='\\s+')
    expected_final = numpy.array([0.0190799, 0.03826254, 0.06080717, 0.101, 0.136, 0.34, 0.457, 0.514, 0.629, 0.638, 0.774, 0.788, 0.08745914, 0.25257575, 0.58544205, 0.01711153, 0.03373885, 0.05287083, 0.07506079, 0.10081573, 1.0, 0.13070334, 0.16539309, 0.20569039, 0.25257575, 0.30725491, 0.37122555, 0.44636843, 0.53507405, 0.64042242, 0.76644378, 0.91850581, 1.10390531, 1.1, 2.0, 2.0, 2.404, 2.86, 3.0, 3.0, 3.705, 4.0, 5.0, 5.96, 6.0, 7.214, 16.0, 17.716, 25.0, 51.0])
    expected_cohn = pandas.DataFrame({'nuncen_above': numpy.array([9.0, 0.0, 18.0, numpy.nan]), 'nobs_below': numpy.array([3.0, 15.0, 32.0, numpy.nan]), 'ncen_equal': numpy.array([3.0, 3.0, 17.0, numpy.nan]), 'prob_exceedance': numpy.array([0.84, 0.36, 0.36, 0])})