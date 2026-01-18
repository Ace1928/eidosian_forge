from io import StringIO
import pytest
from pandas import read_sas
import pandas._testing as tm
def test_sas_buffer_format(self):
    b = StringIO('')
    msg = 'If this is a buffer object rather than a string name, you must specify a format string'
    with pytest.raises(ValueError, match=msg):
        read_sas(b)