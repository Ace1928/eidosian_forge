from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_type_to_pandas_dtype_check_import():
    test_util.invoke_script('arrow_7980.py')