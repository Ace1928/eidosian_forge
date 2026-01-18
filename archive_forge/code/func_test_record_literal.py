from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
def test_record_literal():
    assert R is Record
    assert R['a':'int32'] == R([('a', 'int32')])
    assert R['a':'int32', 'b':'int64'] == R([('a', 'int32'), ('b', 'int64')])