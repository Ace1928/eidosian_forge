import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestLreshape:

    def test_pairs(self):
        data = {'birthdt': ['08jan2009', '20dec2008', '30dec2008', '21dec2008', '11jan2009'], 'birthwt': [1766, 3301, 1454, 3139, 4133], 'id': [101, 102, 103, 104, 105], 'sex': ['Male', 'Female', 'Female', 'Female', 'Female'], 'visitdt1': ['11jan2009', '22dec2008', '04jan2009', '29dec2008', '20jan2009'], 'visitdt2': ['21jan2009', np.nan, '22jan2009', '31dec2008', '03feb2009'], 'visitdt3': ['05feb2009', np.nan, np.nan, '02jan2009', '15feb2009'], 'wt1': [1823, 3338, 1549, 3298, 4306], 'wt2': [2011.0, np.nan, 1892.0, 3338.0, 4575.0], 'wt3': [2293.0, np.nan, np.nan, 3377.0, 4805.0]}
        df = DataFrame(data)
        spec = {'visitdt': [f'visitdt{i:d}' for i in range(1, 4)], 'wt': [f'wt{i:d}' for i in range(1, 4)]}
        result = lreshape(df, spec)
        exp_data = {'birthdt': ['08jan2009', '20dec2008', '30dec2008', '21dec2008', '11jan2009', '08jan2009', '30dec2008', '21dec2008', '11jan2009', '08jan2009', '21dec2008', '11jan2009'], 'birthwt': [1766, 3301, 1454, 3139, 4133, 1766, 1454, 3139, 4133, 1766, 3139, 4133], 'id': [101, 102, 103, 104, 105, 101, 103, 104, 105, 101, 104, 105], 'sex': ['Male', 'Female', 'Female', 'Female', 'Female', 'Male', 'Female', 'Female', 'Female', 'Male', 'Female', 'Female'], 'visitdt': ['11jan2009', '22dec2008', '04jan2009', '29dec2008', '20jan2009', '21jan2009', '22jan2009', '31dec2008', '03feb2009', '05feb2009', '02jan2009', '15feb2009'], 'wt': [1823.0, 3338.0, 1549.0, 3298.0, 4306.0, 2011.0, 1892.0, 3338.0, 4575.0, 2293.0, 3377.0, 4805.0]}
        exp = DataFrame(exp_data, columns=result.columns)
        tm.assert_frame_equal(result, exp)
        result = lreshape(df, spec, dropna=False)
        exp_data = {'birthdt': ['08jan2009', '20dec2008', '30dec2008', '21dec2008', '11jan2009', '08jan2009', '20dec2008', '30dec2008', '21dec2008', '11jan2009', '08jan2009', '20dec2008', '30dec2008', '21dec2008', '11jan2009'], 'birthwt': [1766, 3301, 1454, 3139, 4133, 1766, 3301, 1454, 3139, 4133, 1766, 3301, 1454, 3139, 4133], 'id': [101, 102, 103, 104, 105, 101, 102, 103, 104, 105, 101, 102, 103, 104, 105], 'sex': ['Male', 'Female', 'Female', 'Female', 'Female', 'Male', 'Female', 'Female', 'Female', 'Female', 'Male', 'Female', 'Female', 'Female', 'Female'], 'visitdt': ['11jan2009', '22dec2008', '04jan2009', '29dec2008', '20jan2009', '21jan2009', np.nan, '22jan2009', '31dec2008', '03feb2009', '05feb2009', np.nan, np.nan, '02jan2009', '15feb2009'], 'wt': [1823.0, 3338.0, 1549.0, 3298.0, 4306.0, 2011.0, np.nan, 1892.0, 3338.0, 4575.0, 2293.0, np.nan, np.nan, 3377.0, 4805.0]}
        exp = DataFrame(exp_data, columns=result.columns)
        tm.assert_frame_equal(result, exp)
        spec = {'visitdt': [f'visitdt{i:d}' for i in range(1, 3)], 'wt': [f'wt{i:d}' for i in range(1, 4)]}
        msg = 'All column lists must be same length'
        with pytest.raises(ValueError, match=msg):
            lreshape(df, spec)