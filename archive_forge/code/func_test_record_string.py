from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_record_string():
    s = '{name_with_underscores: int32}'
    assert s.replace(' ', '') == str(dshape(s)).replace(' ', '')