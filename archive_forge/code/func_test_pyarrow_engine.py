from io import StringIO
import os
from pathlib import Path
import pytest
from pandas.errors import ParserError
import pandas._testing as tm
from pandas.io.parsers import read_csv
import pandas.io.parsers.readers as parsers
def test_pyarrow_engine(self):
    from pandas.io.parsers.readers import _pyarrow_unsupported as pa_unsupported
    data = '1,2,3,,\n        1,2,3,4,\n        1,2,3,4,5\n        1,2,,,\n        1,2,3,4,'
    for default in pa_unsupported:
        msg = f"The {repr(default)} option is not supported with the 'pyarrow' engine"
        kwargs = {default: object()}
        default_needs_bool = {'warn_bad_lines', 'error_bad_lines'}
        if default == 'dialect':
            kwargs[default] = 'excel'
        elif default in default_needs_bool:
            kwargs[default] = True
        elif default == 'on_bad_lines':
            kwargs[default] = 'warn'
        warn = None
        depr_msg = None
        if 'delim_whitespace' in kwargs:
            depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
            warn = FutureWarning
        if 'verbose' in kwargs:
            depr_msg = "The 'verbose' keyword in pd.read_csv is deprecated"
            warn = FutureWarning
        with pytest.raises(ValueError, match=msg):
            with tm.assert_produces_warning(warn, match=depr_msg):
                read_csv(StringIO(data), engine='pyarrow', **kwargs)