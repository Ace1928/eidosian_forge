from io import StringIO
import os
from pathlib import Path
import pytest
from pandas.errors import ParserError
import pandas._testing as tm
from pandas.io.parsers import read_csv
import pandas.io.parsers.readers as parsers
def test_on_bad_lines_callable_python_or_pyarrow(self, all_parsers):
    sio = StringIO('a,b\n1,2')
    bad_lines_func = lambda x: x
    parser = all_parsers
    if all_parsers.engine not in ['python', 'pyarrow']:
        msg = "on_bad_line can only be a callable function if engine='python' or 'pyarrow'"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(sio, on_bad_lines=bad_lines_func)
    else:
        parser.read_csv(sio, on_bad_lines=bad_lines_func)