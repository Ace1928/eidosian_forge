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
class TestCommonIOCapabilities:
    data1 = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'

    def test_expand_user(self):
        filename = '~/sometest'
        expanded_name = icom._expand_user(filename)
        assert expanded_name != filename
        assert os.path.isabs(expanded_name)
        assert os.path.expanduser(filename) == expanded_name

    def test_expand_user_normal_path(self):
        filename = '/somefolder/sometest'
        expanded_name = icom._expand_user(filename)
        assert expanded_name == filename
        assert os.path.expanduser(filename) == expanded_name

    def test_stringify_path_pathlib(self):
        rel_path = icom.stringify_path(Path('.'))
        assert rel_path == '.'
        redundant_path = icom.stringify_path(Path('foo//bar'))
        assert redundant_path == os.path.join('foo', 'bar')

    @td.skip_if_no('py.path')
    def test_stringify_path_localpath(self):
        path = os.path.join('foo', 'bar')
        abs_path = os.path.abspath(path)
        lpath = LocalPath(path)
        assert icom.stringify_path(lpath) == abs_path

    def test_stringify_path_fspath(self):
        p = CustomFSPath('foo/bar.csv')
        result = icom.stringify_path(p)
        assert result == 'foo/bar.csv'

    def test_stringify_file_and_path_like(self):
        fsspec = pytest.importorskip('fsspec')
        with tm.ensure_clean() as path:
            with fsspec.open(f'file://{path}', mode='wb') as fsspec_obj:
                assert fsspec_obj == icom.stringify_path(fsspec_obj)

    @pytest.mark.parametrize('path_type', path_types)
    def test_infer_compression_from_path(self, compression_format, path_type):
        extension, expected = compression_format
        path = path_type('foo/bar.csv' + extension)
        compression = icom.infer_compression(path, compression='infer')
        assert compression == expected

    @pytest.mark.parametrize('path_type', [str, CustomFSPath, Path])
    def test_get_handle_with_path(self, path_type):
        with tempfile.TemporaryDirectory(dir=Path.home()) as tmp:
            filename = path_type('~/' + Path(tmp).name + '/sometest')
            with icom.get_handle(filename, 'w') as handles:
                assert Path(handles.handle.name).is_absolute()
                assert os.path.expanduser(filename) == handles.handle.name

    def test_get_handle_with_buffer(self):
        with StringIO() as input_buffer:
            with icom.get_handle(input_buffer, 'r') as handles:
                assert handles.handle == input_buffer
            assert not input_buffer.closed
        assert input_buffer.closed

    def test_bytesiowrapper_returns_correct_bytes(self):
        data = 'a,b,c\n1,2,3\n©,®,®\nLook,a snake,🐍'
        with icom.get_handle(StringIO(data), 'rb', is_text=False) as handles:
            result = b''
            chunksize = 5
            while True:
                chunk = handles.handle.read(chunksize)
                assert len(chunk) <= chunksize
                if len(chunk) < chunksize:
                    assert len(handles.handle.read()) == 0
                    result += chunk
                    break
                result += chunk
            assert result == data.encode('utf-8')

    def test_get_handle_pyarrow_compat(self):
        pa_csv = pytest.importorskip('pyarrow.csv')
        data = 'a,b,c\n1,2,3\n©,®,®\nLook,a snake,🐍'
        expected = pd.DataFrame({'a': ['1', '©', 'Look'], 'b': ['2', '®', 'a snake'], 'c': ['3', '®', '🐍']})
        s = StringIO(data)
        with icom.get_handle(s, 'rb', is_text=False) as handles:
            df = pa_csv.read_csv(handles.handle).to_pandas()
            tm.assert_frame_equal(df, expected)
            assert not s.closed

    def test_iterator(self):
        with pd.read_csv(StringIO(self.data1), chunksize=1) as reader:
            result = pd.concat(reader, ignore_index=True)
        expected = pd.read_csv(StringIO(self.data1))
        tm.assert_frame_equal(result, expected)
        with pd.read_csv(StringIO(self.data1), chunksize=1) as it:
            first = next(it)
            tm.assert_frame_equal(first, expected.iloc[[0]])
            tm.assert_frame_equal(pd.concat(it), expected.iloc[1:])

    @pytest.mark.parametrize('reader, module, error_class, fn_ext', [(pd.read_csv, 'os', FileNotFoundError, 'csv'), (pd.read_fwf, 'os', FileNotFoundError, 'txt'), (pd.read_excel, 'xlrd', FileNotFoundError, 'xlsx'), (pd.read_feather, 'pyarrow', OSError, 'feather'), (pd.read_hdf, 'tables', FileNotFoundError, 'h5'), (pd.read_stata, 'os', FileNotFoundError, 'dta'), (pd.read_sas, 'os', FileNotFoundError, 'sas7bdat'), (pd.read_json, 'os', FileNotFoundError, 'json'), (pd.read_pickle, 'os', FileNotFoundError, 'pickle')])
    def test_read_non_existent(self, reader, module, error_class, fn_ext):
        pytest.importorskip(module)
        path = os.path.join(HERE, 'data', 'does_not_exist.' + fn_ext)
        msg1 = f"File (b')?.+does_not_exist\\.{fn_ext}'? does not exist"
        msg2 = f"\\[Errno 2\\] No such file or directory: '.+does_not_exist\\.{fn_ext}'"
        msg3 = 'Expected object or value'
        msg4 = 'path_or_buf needs to be a string file path or file-like'
        msg5 = f"\\[Errno 2\\] File .+does_not_exist\\.{fn_ext} does not exist: '.+does_not_exist\\.{fn_ext}'"
        msg6 = f"\\[Errno 2\\] 没有那个文件或目录: '.+does_not_exist\\.{fn_ext}'"
        msg7 = f"\\[Errno 2\\] File o directory non esistente: '.+does_not_exist\\.{fn_ext}'"
        msg8 = f'Failed to open local file.+does_not_exist\\.{fn_ext}'
        with pytest.raises(error_class, match=f'({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})'):
            reader(path)

    @pytest.mark.parametrize('method, module, error_class, fn_ext', [(pd.DataFrame.to_csv, 'os', OSError, 'csv'), (pd.DataFrame.to_html, 'os', OSError, 'html'), (pd.DataFrame.to_excel, 'xlrd', OSError, 'xlsx'), (pd.DataFrame.to_feather, 'pyarrow', OSError, 'feather'), (pd.DataFrame.to_parquet, 'pyarrow', OSError, 'parquet'), (pd.DataFrame.to_stata, 'os', OSError, 'dta'), (pd.DataFrame.to_json, 'os', OSError, 'json'), (pd.DataFrame.to_pickle, 'os', OSError, 'pickle')])
    def test_write_missing_parent_directory(self, method, module, error_class, fn_ext):
        pytest.importorskip(module)
        dummy_frame = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [3, 4, 5]})
        path = os.path.join(HERE, 'data', 'missing_folder', 'does_not_exist.' + fn_ext)
        with pytest.raises(error_class, match='Cannot save file into a non-existent directory: .*missing_folder'):
            method(dummy_frame, path)

    @pytest.mark.parametrize('reader, module, error_class, fn_ext', [(pd.read_csv, 'os', FileNotFoundError, 'csv'), (pd.read_table, 'os', FileNotFoundError, 'csv'), (pd.read_fwf, 'os', FileNotFoundError, 'txt'), (pd.read_excel, 'xlrd', FileNotFoundError, 'xlsx'), (pd.read_feather, 'pyarrow', OSError, 'feather'), (pd.read_hdf, 'tables', FileNotFoundError, 'h5'), (pd.read_stata, 'os', FileNotFoundError, 'dta'), (pd.read_sas, 'os', FileNotFoundError, 'sas7bdat'), (pd.read_json, 'os', FileNotFoundError, 'json'), (pd.read_pickle, 'os', FileNotFoundError, 'pickle')])
    def test_read_expands_user_home_dir(self, reader, module, error_class, fn_ext, monkeypatch):
        pytest.importorskip(module)
        path = os.path.join('~', 'does_not_exist.' + fn_ext)
        monkeypatch.setattr(icom, '_expand_user', lambda x: os.path.join('foo', x))
        msg1 = f"File (b')?.+does_not_exist\\.{fn_ext}'? does not exist"
        msg2 = f"\\[Errno 2\\] No such file or directory: '.+does_not_exist\\.{fn_ext}'"
        msg3 = "Unexpected character found when decoding 'false'"
        msg4 = 'path_or_buf needs to be a string file path or file-like'
        msg5 = f"\\[Errno 2\\] File .+does_not_exist\\.{fn_ext} does not exist: '.+does_not_exist\\.{fn_ext}'"
        msg6 = f"\\[Errno 2\\] 没有那个文件或目录: '.+does_not_exist\\.{fn_ext}'"
        msg7 = f"\\[Errno 2\\] File o directory non esistente: '.+does_not_exist\\.{fn_ext}'"
        msg8 = f'Failed to open local file.+does_not_exist\\.{fn_ext}'
        with pytest.raises(error_class, match=f'({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})'):
            reader(path)

    @pytest.mark.parametrize('reader, module, path', [(pd.read_csv, 'os', ('io', 'data', 'csv', 'iris.csv')), (pd.read_table, 'os', ('io', 'data', 'csv', 'iris.csv')), (pd.read_fwf, 'os', ('io', 'data', 'fixed_width', 'fixed_width_format.txt')), (pd.read_excel, 'xlrd', ('io', 'data', 'excel', 'test1.xlsx')), (pd.read_feather, 'pyarrow', ('io', 'data', 'feather', 'feather-0_3_1.feather')), (pd.read_hdf, 'tables', ('io', 'data', 'legacy_hdf', 'datetimetz_object.h5')), (pd.read_stata, 'os', ('io', 'data', 'stata', 'stata10_115.dta')), (pd.read_sas, 'os', ('io', 'sas', 'data', 'test1.sas7bdat')), (pd.read_json, 'os', ('io', 'json', 'data', 'tsframe_v012.json')), (pd.read_pickle, 'os', ('io', 'data', 'pickle', 'categorical.0.25.0.pickle'))])
    def test_read_fspath_all(self, reader, module, path, datapath):
        pytest.importorskip(module)
        path = datapath(*path)
        mypath = CustomFSPath(path)
        result = reader(mypath)
        expected = reader(path)
        if path.endswith('.pickle'):
            tm.assert_categorical_equal(result, expected)
        else:
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('writer_name, writer_kwargs, module', [('to_csv', {}, 'os'), ('to_excel', {'engine': 'openpyxl'}, 'openpyxl'), ('to_feather', {}, 'pyarrow'), ('to_html', {}, 'os'), ('to_json', {}, 'os'), ('to_latex', {}, 'os'), ('to_pickle', {}, 'os'), ('to_stata', {'time_stamp': pd.to_datetime('2019-01-01 00:00')}, 'os')])
    def test_write_fspath_all(self, writer_name, writer_kwargs, module):
        if writer_name in ['to_latex']:
            pytest.importorskip('jinja2')
        p1 = tm.ensure_clean('string')
        p2 = tm.ensure_clean('fspath')
        df = pd.DataFrame({'A': [1, 2]})
        with p1 as string, p2 as fspath:
            pytest.importorskip(module)
            mypath = CustomFSPath(fspath)
            writer = getattr(df, writer_name)
            writer(string, **writer_kwargs)
            writer(mypath, **writer_kwargs)
            with open(string, 'rb') as f_str, open(fspath, 'rb') as f_path:
                if writer_name == 'to_excel':
                    result = pd.read_excel(f_str, **writer_kwargs)
                    expected = pd.read_excel(f_path, **writer_kwargs)
                    tm.assert_frame_equal(result, expected)
                else:
                    result = f_str.read()
                    expected = f_path.read()
                    assert result == expected

    def test_write_fspath_hdf5(self):
        pytest.importorskip('tables')
        df = pd.DataFrame({'A': [1, 2]})
        p1 = tm.ensure_clean('string')
        p2 = tm.ensure_clean('fspath')
        with p1 as string, p2 as fspath:
            mypath = CustomFSPath(fspath)
            df.to_hdf(mypath, key='bar')
            df.to_hdf(string, key='bar')
            result = pd.read_hdf(fspath, key='bar')
            expected = pd.read_hdf(string, key='bar')
        tm.assert_frame_equal(result, expected)