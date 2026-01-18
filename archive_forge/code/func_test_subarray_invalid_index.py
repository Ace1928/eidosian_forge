from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_subarray_invalid_index():
    with pytest.raises(IndexError):
        dshape('1 * 2 * 3 * int32').subarray(42)