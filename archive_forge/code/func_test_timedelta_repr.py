from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
@pytest.mark.parametrize('unit', _units)
def test_timedelta_repr(unit):
    assert repr(TimeDelta(unit=unit)) == 'TimeDelta(unit=%r)' % unit