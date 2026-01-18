from __future__ import annotations
import traceback
from contextlib import contextmanager
import pytest
import dask
from dask.utils import shorten_traceback
@pytest.mark.parametrize('regexes,expect', [(None, ['test_shorten_traceback', 'f3', 'f2', 'f1']), ([], ['test_shorten_traceback', 'f3', 'f2', 'f1']), (['nomatch'], ['test_shorten_traceback', 'f3', 'f2', 'f1']), (['.*'], ['test_shorten_traceback', 'f1']), (['tests'], ['test_shorten_traceback', 'f1'])])
def test_shorten_traceback(regexes, expect):
    """
    See also
    --------
    test_distributed.py::test_shorten_traceback_excepthook
    test_distributed.py::test_shorten_traceback_ipython
    """
    with dask.config.set({'admin.traceback.shorten': regexes}):
        with assert_tb_levels(expect):
            with shorten_traceback():
                f3()