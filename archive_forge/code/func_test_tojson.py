from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import json
import pytest
from petl.test.helpers import ieq
from petl import fromjson, fromdicts, tojson, tojsonarrays
def test_tojson():
    table = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2))
    f = NamedTemporaryFile(delete=False, mode='r')
    tojson(table, f.name)
    result = json.load(f)
    assert len(result) == 3
    assert result[0]['foo'] == 'a'
    assert result[0]['bar'] == 1
    assert result[1]['foo'] == 'b'
    assert result[1]['bar'] == 2
    assert result[2]['foo'] == 'c'
    assert result[2]['bar'] == 2