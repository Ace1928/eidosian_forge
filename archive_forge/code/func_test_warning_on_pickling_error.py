import functools
from pickle import PicklingError
import time
import pytest
from joblib.testing import parametrize, timeout
from joblib.test.common import with_multiprocessing
from joblib.backports import concurrency_safe_rename
from joblib import Parallel, delayed
from joblib._store_backends import (
def test_warning_on_pickling_error(tmpdir):

    class UnpicklableObject(object):

        def __reduce__(self):
            raise PicklingError('not picklable')
    backend = FileSystemStoreBackend()
    backend.location = tmpdir.join('test_warning_on_pickling_error').strpath
    backend.compress = None
    with pytest.warns(FutureWarning, match='not picklable'):
        backend.dump_item('testpath', UnpicklableObject())