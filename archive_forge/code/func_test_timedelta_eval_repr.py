from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_timedelta_eval_repr():
    assert eval(repr(dshape('timedelta'))) == dshape('timedelta')
    assert eval(repr(dshape('timedelta[unit="ms"]'))) == dshape('timedelta[unit="ms"]')