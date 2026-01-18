from collections import OrderedDict
from itertools import starmap
from types import MappingProxyType
from warnings import catch_warnings, simplefilter
import numpy as np
import pytest
from datashader.datashape.discovery import (
from datashader.datashape.coretypes import (
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape import dshape
from datetime import date, time, datetime, timedelta
def test_unite_records():
    assert discover([{'name': 'Alice', 'balance': 100}, {'name': 'Bob', 'balance': ''}]) == 2 * Record([['balance', Option(int64)], ['name', string]])
    assert discover([{'name': 'Alice', 's': 'foo'}, {'name': 'Bob', 's': None}]) == 2 * Record([['name', string], ['s', Option(string)]])
    assert discover([{'name': 'Alice', 's': 'foo', 'f': 1.0}, {'name': 'Bob', 's': None, 'f': None}]) == 2 * Record([['f', Option(float64)], ['name', string], ['s', Option(string)]])