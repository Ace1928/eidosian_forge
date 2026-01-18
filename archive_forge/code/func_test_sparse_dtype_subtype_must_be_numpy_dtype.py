import re
import warnings
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
def test_sparse_dtype_subtype_must_be_numpy_dtype():
    msg = 'SparseDtype subtype must be a numpy dtype'
    with pytest.raises(TypeError, match=msg):
        SparseDtype('category', fill_value='c')