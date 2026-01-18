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
@pytest.mark.parametrize('method, module, error_class, fn_ext', [(pd.DataFrame.to_csv, 'os', OSError, 'csv'), (pd.DataFrame.to_html, 'os', OSError, 'html'), (pd.DataFrame.to_excel, 'xlrd', OSError, 'xlsx'), (pd.DataFrame.to_feather, 'pyarrow', OSError, 'feather'), (pd.DataFrame.to_parquet, 'pyarrow', OSError, 'parquet'), (pd.DataFrame.to_stata, 'os', OSError, 'dta'), (pd.DataFrame.to_json, 'os', OSError, 'json'), (pd.DataFrame.to_pickle, 'os', OSError, 'pickle')])
def test_write_missing_parent_directory(self, method, module, error_class, fn_ext):
    pytest.importorskip(module)
    dummy_frame = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [3, 4, 5]})
    path = os.path.join(HERE, 'data', 'missing_folder', 'does_not_exist.' + fn_ext)
    with pytest.raises(error_class, match='Cannot save file into a non-existent directory: .*missing_folder'):
        method(dummy_frame, path)