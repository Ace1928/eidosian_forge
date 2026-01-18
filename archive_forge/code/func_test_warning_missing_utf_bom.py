import codecs
import errno
from functools import partial
from io import (
import mmap
import os
from pathlib import Path
import pickle
import tempfile
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom
@pytest.mark.parametrize('encoding', ['utf-16', 'utf-32'])
@pytest.mark.parametrize('compression_', ['bz2', 'xz'])
def test_warning_missing_utf_bom(self, encoding, compression_):
    """
        bz2 and xz do not write the byte order mark (BOM) for utf-16/32.

        https://stackoverflow.com/questions/55171439

        GH 35681
        """
    df = pd.DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD'), dtype=object), index=pd.Index([f'i-{i}' for i in range(30)], dtype=object))
    with tm.ensure_clean() as path:
        with tm.assert_produces_warning(UnicodeWarning):
            df.to_csv(path, compression=compression_, encoding=encoding)
        msg = 'UTF-\\d+ stream does not start with BOM'
        with pytest.raises(UnicodeError, match=msg):
            pd.read_csv(path, compression=compression_, encoding=encoding)