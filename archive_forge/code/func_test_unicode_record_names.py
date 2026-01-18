from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
@pytest.mark.parametrize(['names', 'typ'], [(['foo', b'\xc4\x87'.decode('utf8')], str), (['foo', 'bar'], str), (list(u'ab'), str)])
def test_unicode_record_names(names, typ):
    types = [int64, float64]
    record = Record(list(zip(names, types)))
    string_type, = set(map(type, record.names))
    assert record.names == names
    assert record.types == types
    assert all((isinstance(s, string_type) for s in record.names))