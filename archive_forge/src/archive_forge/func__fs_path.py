import dataclasses
import glob as py_glob
import io
import os
import os.path
import sys
import tempfile
from tensorboard.compat.tensorflow_stub import compat, errors
def _fs_path(self, filename):
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8')
    self._validate_path(filename)
    fs, path = fsspec.core.url_to_fs(filename)
    return (fs, path)