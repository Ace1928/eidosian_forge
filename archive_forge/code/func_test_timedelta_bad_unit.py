from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_timedelta_bad_unit():
    with pytest.raises(ValueError):
        dshape('timedelta[unit="foo"]')