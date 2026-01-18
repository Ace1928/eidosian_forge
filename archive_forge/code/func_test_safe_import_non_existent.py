import sys
import types
import pytest
import pandas.util._test_decorators as td
@pytest.mark.parametrize('name', ['foo', 'hello123'])
def test_safe_import_non_existent(name):
    assert not td.safe_import(name)