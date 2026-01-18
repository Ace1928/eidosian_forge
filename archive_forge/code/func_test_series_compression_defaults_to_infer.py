import gzip
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap
import time
import zipfile
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom
@pytest.mark.parametrize('write_method,write_kwargs,read_method,read_kwargs', [('to_csv', {'index': False, 'header': True}, pd.read_csv, {'squeeze': True}), ('to_json', {}, pd.read_json, {'typ': 'series'}), ('to_pickle', {}, pd.read_pickle, {})])
def test_series_compression_defaults_to_infer(write_method, write_kwargs, read_method, read_kwargs, compression_only, compression_to_extension):
    input = pd.Series([0, 5, -2, 10], name='X')
    extension = compression_to_extension[compression_only]
    with tm.ensure_clean('compressed' + extension) as path:
        getattr(input, write_method)(path, **write_kwargs)
        if 'squeeze' in read_kwargs:
            kwargs = read_kwargs.copy()
            del kwargs['squeeze']
            output = read_method(path, compression=compression_only, **kwargs).squeeze('columns')
        else:
            output = read_method(path, compression=compression_only, **read_kwargs)
    tm.assert_series_equal(output, input, check_names=False)