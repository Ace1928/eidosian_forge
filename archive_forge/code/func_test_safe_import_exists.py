import sys
import types
import pytest
import pandas.util._test_decorators as td
def test_safe_import_exists():
    assert td.safe_import('pandas')