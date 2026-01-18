from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_record_with_unicode_name_as_numpy_dtype():
    r = Record([(str('a'), 'int32')])
    assert r.to_numpy_dtype() == np.dtype([('a', 'i4')])