from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_record_subshape_integer_index():
    ds = DataShape(Record([('a', 'int32')]))
    assert ds.subshape[0] == int32