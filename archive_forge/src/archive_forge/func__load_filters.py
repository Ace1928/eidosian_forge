from _pydev_runfiles import pydev_runfiles_xml_rpc
import pickle
import zlib
import base64
import os
from pydevd_file_utils import canonical_normalized_path
import pytest
import sys
import time
from pathlib import Path
def _load_filters():
    global py_test_accept_filter
    if py_test_accept_filter is None:
        py_test_accept_filter = os.environ.get('PYDEV_PYTEST_SKIP')
        if py_test_accept_filter:
            py_test_accept_filter = pickle.loads(zlib.decompress(base64.b64decode(py_test_accept_filter)))
            new_dct = {}
            for filename, value in py_test_accept_filter.items():
                new_dct[canonical_normalized_path(str(Path(filename).resolve()))] = value
            py_test_accept_filter.update(new_dct)
        else:
            py_test_accept_filter = {}