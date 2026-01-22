from statsmodels.compat.python import lrange
from io import BytesIO
import os
import pathlib
import tempfile
from numpy.testing import assert_equal
from statsmodels.iolib.smpickle import load_pickle, save_pickle
class SubPath:

    def __init__(self, path):
        self._path = pathlib.Path(path)

    def open(self, mode='r', buffering=-1, encoding=None, errors=None, newline=None):
        return self._path.open(mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline)