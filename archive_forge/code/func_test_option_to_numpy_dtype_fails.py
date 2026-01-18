from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
@pytest.mark.xfail(raises=TypeError, reason='NumPy has no notion of missing for types other than timedelta, datetime, and date')
@pytest.mark.parametrize('base', [int32, float64, Record([('a', uint32)])])
def test_option_to_numpy_dtype_fails(base):
    Option(base).to_numpy_dtype()