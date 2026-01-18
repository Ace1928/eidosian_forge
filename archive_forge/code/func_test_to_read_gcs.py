from io import BytesIO
import os
import pathlib
import tarfile
import zipfile
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
@pytest.mark.single_cpu
@pytest.mark.parametrize('format', ['csv', 'json', 'parquet', 'excel', 'markdown'])
def test_to_read_gcs(gcs_buffer, format, monkeypatch, capsys):
    """
    Test that many to/read functions support GCS.

    GH 33987
    """
    df1 = DataFrame({'int': [1, 3], 'float': [2.0, np.nan], 'str': ['t', 's'], 'dt': date_range('2018-06-18', periods=2)})
    path = f'gs://test/test.{format}'
    if format == 'csv':
        df1.to_csv(path, index=True)
        df2 = read_csv(path, parse_dates=['dt'], index_col=0)
    elif format == 'excel':
        path = 'gs://test/test.xlsx'
        df1.to_excel(path)
        df2 = read_excel(path, parse_dates=['dt'], index_col=0)
    elif format == 'json':
        df1.to_json(path)
        df2 = read_json(path, convert_dates=['dt'])
    elif format == 'parquet':
        pytest.importorskip('pyarrow')
        pa_fs = pytest.importorskip('pyarrow.fs')

        class MockFileSystem(pa_fs.FileSystem):

            @staticmethod
            def from_uri(path):
                print('Using pyarrow filesystem')
                to_local = pathlib.Path(path.replace('gs://', '')).absolute().as_uri()
                return pa_fs.LocalFileSystem(to_local)
        with monkeypatch.context() as m:
            m.setattr(pa_fs, 'FileSystem', MockFileSystem)
            df1.to_parquet(path)
            df2 = read_parquet(path)
        captured = capsys.readouterr()
        assert captured.out == 'Using pyarrow filesystem\nUsing pyarrow filesystem\n'
    elif format == 'markdown':
        pytest.importorskip('tabulate')
        df1.to_markdown(path)
        df2 = df1
    tm.assert_frame_equal(df1, df2)