from statsmodels.compat.python import lrange
from io import BytesIO
import os
import pathlib
import tempfile
from numpy.testing import assert_equal
from statsmodels.iolib.smpickle import load_pickle, save_pickle
def test_pickle_supports_open():
    tmpdir = tempfile.mkdtemp(prefix='pickle')
    a = lrange(10)

    class SubPath:

        def __init__(self, path):
            self._path = pathlib.Path(path)

        def open(self, mode='r', buffering=-1, encoding=None, errors=None, newline=None):
            return self._path.open(mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline)
    path_pathlib = SubPath(tmpdir + os.pathsep + 'res2.pkl')
    save_pickle(a, path_pathlib)
    c = load_pickle(path_pathlib)
    assert_equal(a, c)