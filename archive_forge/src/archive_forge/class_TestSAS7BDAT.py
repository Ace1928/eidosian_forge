import contextlib
from datetime import datetime
import io
import os
from pathlib import Path
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import EmptyDataError
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sas7bdat import SAS7BDATReader
class TestSAS7BDAT:

    @pytest.mark.slow
    def test_from_file(self, dirpath, data_test_ix):
        expected, test_ix = data_test_ix
        for k in test_ix:
            fname = os.path.join(dirpath, f'test{k}.sas7bdat')
            df = pd.read_sas(fname, encoding='utf-8')
            tm.assert_frame_equal(df, expected)

    @pytest.mark.slow
    def test_from_buffer(self, dirpath, data_test_ix):
        expected, test_ix = data_test_ix
        for k in test_ix:
            fname = os.path.join(dirpath, f'test{k}.sas7bdat')
            with open(fname, 'rb') as f:
                byts = f.read()
            buf = io.BytesIO(byts)
            with pd.read_sas(buf, format='sas7bdat', iterator=True, encoding='utf-8') as rdr:
                df = rdr.read()
            tm.assert_frame_equal(df, expected)

    @pytest.mark.slow
    def test_from_iterator(self, dirpath, data_test_ix):
        expected, test_ix = data_test_ix
        for k in test_ix:
            fname = os.path.join(dirpath, f'test{k}.sas7bdat')
            with pd.read_sas(fname, iterator=True, encoding='utf-8') as rdr:
                df = rdr.read(2)
                tm.assert_frame_equal(df, expected.iloc[0:2, :])
                df = rdr.read(3)
                tm.assert_frame_equal(df, expected.iloc[2:5, :])

    @pytest.mark.slow
    def test_path_pathlib(self, dirpath, data_test_ix):
        expected, test_ix = data_test_ix
        for k in test_ix:
            fname = Path(os.path.join(dirpath, f'test{k}.sas7bdat'))
            df = pd.read_sas(fname, encoding='utf-8')
            tm.assert_frame_equal(df, expected)

    @td.skip_if_no('py.path')
    @pytest.mark.slow
    def test_path_localpath(self, dirpath, data_test_ix):
        from py.path import local as LocalPath
        expected, test_ix = data_test_ix
        for k in test_ix:
            fname = LocalPath(os.path.join(dirpath, f'test{k}.sas7bdat'))
            df = pd.read_sas(fname, encoding='utf-8')
            tm.assert_frame_equal(df, expected)

    @pytest.mark.slow
    @pytest.mark.parametrize('chunksize', (3, 5, 10, 11))
    @pytest.mark.parametrize('k', range(1, 17))
    def test_iterator_loop(self, dirpath, k, chunksize):
        fname = os.path.join(dirpath, f'test{k}.sas7bdat')
        with pd.read_sas(fname, chunksize=chunksize, encoding='utf-8') as rdr:
            y = 0
            for x in rdr:
                y += x.shape[0]
        assert y == rdr.row_count

    def test_iterator_read_too_much(self, dirpath):
        fname = os.path.join(dirpath, 'test1.sas7bdat')
        with pd.read_sas(fname, format='sas7bdat', iterator=True, encoding='utf-8') as rdr:
            d1 = rdr.read(rdr.row_count + 20)
        with pd.read_sas(fname, iterator=True, encoding='utf-8') as rdr:
            d2 = rdr.read(rdr.row_count + 20)
        tm.assert_frame_equal(d1, d2)