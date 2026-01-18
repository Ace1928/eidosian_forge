from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_string_from_CType_classmethod(self):
    assert CType.from_numpy_dtype(np.dtype('S7')) == String(7, 'A')