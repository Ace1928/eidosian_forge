from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_subshape():
    ds = dshape('5 * 3 * float32')
    assert ds.subshape[2:] == dshape('3 * 3 * float32')
    ds = dshape('5 * 3 * float32')
    assert ds.subshape[::2] == dshape('3 * 3 * float32')