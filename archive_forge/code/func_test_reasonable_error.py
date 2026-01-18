import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
def test_reasonable_error(monkeypatch, cleared_fs):
    from fsspec.registry import known_implementations
    with pytest.raises(ValueError, match='nosuchprotocol'):
        read_csv('nosuchprotocol://test/test.csv')
    err_msg = 'test error message'
    monkeypatch.setitem(known_implementations, 'couldexist', {'class': 'unimportable.CouldExist', 'err': err_msg})
    with pytest.raises(ImportError, match=err_msg):
        read_csv('couldexist://test/test.csv')