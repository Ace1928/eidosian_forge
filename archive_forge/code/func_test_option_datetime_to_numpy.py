from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_option_datetime_to_numpy():
    assert Option(DateTime()).to_numpy_dtype() == np.dtype('datetime64[us]')