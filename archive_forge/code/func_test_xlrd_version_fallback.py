import sys
import types
import pytest
from pandas.compat._optional import (
import pandas._testing as tm
def test_xlrd_version_fallback():
    pytest.importorskip('xlrd')
    import_optional_dependency('xlrd')