from datetime import datetime, timezone, timedelta
import gzip
import os
import pathlib
import subprocess
import sys
import pytest
import weakref
import pyarrow as pa
from pyarrow.tests.test_io import assert_file_not_found
from pyarrow.tests.util import (_filesystem_uri, ProxyHandler,
from pyarrow.fs import (FileType, FileInfo, FileSelector, FileSystem,
class DummyHandler(FileSystemHandler):

    def __init__(self, value=42):
        self._value = value

    def __eq__(self, other):
        if isinstance(other, FileSystemHandler):
            return self._value == other._value
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, FileSystemHandler):
            return self._value != other._value
        return NotImplemented

    def get_type_name(self):
        return 'dummy'

    def normalize_path(self, path):
        return path

    def get_file_info(self, paths):
        info = []
        for path in paths:
            if 'file' in path:
                info.append(FileInfo(path, FileType.File))
            elif 'dir' in path:
                info.append(FileInfo(path, FileType.Directory))
            elif 'notfound' in path:
                info.append(FileInfo(path, FileType.NotFound))
            elif 'badtype' in path:
                info.append(object())
            else:
                raise IOError
        return info

    def get_file_info_selector(self, selector):
        if selector.base_dir != 'somedir':
            if selector.allow_not_found:
                return []
            else:
                raise FileNotFoundError(selector.base_dir)
        infos = [FileInfo('somedir/file1', FileType.File, size=123), FileInfo('somedir/subdir1', FileType.Directory)]
        if selector.recursive:
            infos += [FileInfo('somedir/subdir1/file2', FileType.File, size=456)]
        return infos

    def create_dir(self, path, recursive):
        if path == 'recursive':
            assert recursive is True
        elif path == 'non-recursive':
            assert recursive is False
        else:
            raise IOError

    def delete_dir(self, path):
        assert path == 'delete_dir'

    def delete_dir_contents(self, path, missing_dir_ok):
        if not path.strip('/'):
            raise ValueError
        assert path == 'delete_dir_contents'

    def delete_root_dir_contents(self):
        pass

    def delete_file(self, path):
        assert path == 'delete_file'

    def move(self, src, dest):
        assert src == 'move_from'
        assert dest == 'move_to'

    def copy_file(self, src, dest):
        assert src == 'copy_file_from'
        assert dest == 'copy_file_to'

    def open_input_stream(self, path):
        if 'notfound' in path:
            raise FileNotFoundError(path)
        data = '{0}:input_stream'.format(path).encode('utf8')
        return pa.BufferReader(data)

    def open_input_file(self, path):
        if 'notfound' in path:
            raise FileNotFoundError(path)
        data = '{0}:input_file'.format(path).encode('utf8')
        return pa.BufferReader(data)

    def open_output_stream(self, path, metadata):
        if 'notfound' in path:
            raise FileNotFoundError(path)
        return pa.BufferOutputStream()

    def open_append_stream(self, path, metadata):
        if 'notfound' in path:
            raise FileNotFoundError(path)
        return pa.BufferOutputStream()