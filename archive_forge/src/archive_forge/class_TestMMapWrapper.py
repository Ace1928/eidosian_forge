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
class TestMMapWrapper:

    def test_constructor_bad_file(self, mmap_file):
        non_file = StringIO('I am not a file')
        non_file.fileno = lambda: -1
        if is_platform_windows():
            msg = 'The parameter is incorrect'
            err = OSError
        else:
            msg = '[Errno 22]'
            err = mmap.error
        with pytest.raises(err, match=msg):
            icom._maybe_memory_map(non_file, True)
        with open(mmap_file, encoding='utf-8') as target:
            pass
        msg = 'I/O operation on closed file'
        with pytest.raises(ValueError, match=msg):
            icom._maybe_memory_map(target, True)

    def test_next(self, mmap_file):
        with open(mmap_file, encoding='utf-8') as target:
            lines = target.readlines()
            with icom.get_handle(target, 'r', is_text=True, memory_map=True) as wrappers:
                wrapper = wrappers.handle
                assert isinstance(wrapper.buffer.buffer, mmap.mmap)
                for line in lines:
                    next_line = next(wrapper)
                    assert next_line.strip() == line.strip()
                with pytest.raises(StopIteration, match='^$'):
                    next(wrapper)

    def test_unknown_engine(self):
        with tm.ensure_clean() as path:
            df = pd.DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD'), dtype=object), index=pd.Index([f'i-{i}' for i in range(30)], dtype=object))
            df.to_csv(path)
            with pytest.raises(ValueError, match='Unknown engine'):
                pd.read_csv(path, engine='pyt')

    def test_binary_mode(self):
        """
        'encoding' shouldn't be passed to 'open' in binary mode.

        GH 35058
        """
        with tm.ensure_clean() as path:
            df = pd.DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD'), dtype=object), index=pd.Index([f'i-{i}' for i in range(30)], dtype=object))
            df.to_csv(path, mode='w+b')
            tm.assert_frame_equal(df, pd.read_csv(path, index_col=0))

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