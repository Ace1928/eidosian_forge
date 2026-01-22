import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
class TestUsingCompiledIfAvailable(tests.TestCase):
    """Check that any compiled functions that are available are the default.

    It is possible to have typos, etc in the import line, such that
    _dirstate_helpers_pyx is actually available, but the compiled functions are
    not being used.
    """

    def test_bisect_dirblock(self):
        if compiled_dirstate_helpers_feature.available():
            from .._dirstate_helpers_pyx import bisect_dirblock
        else:
            from .._dirstate_helpers_py import bisect_dirblock
        self.assertIs(bisect_dirblock, dirstate.bisect_dirblock)

    def test__bisect_path_left(self):
        if compiled_dirstate_helpers_feature.available():
            from .._dirstate_helpers_pyx import _bisect_path_left
        else:
            from .._dirstate_helpers_py import _bisect_path_left
        self.assertIs(_bisect_path_left, dirstate._bisect_path_left)

    def test__bisect_path_right(self):
        if compiled_dirstate_helpers_feature.available():
            from .._dirstate_helpers_pyx import _bisect_path_right
        else:
            from .._dirstate_helpers_py import _bisect_path_right
        self.assertIs(_bisect_path_right, dirstate._bisect_path_right)

    def test_lt_by_dirs(self):
        if compiled_dirstate_helpers_feature.available():
            from .._dirstate_helpers_pyx import lt_by_dirs
        else:
            from .._dirstate_helpers_py import lt_by_dirs
        self.assertIs(lt_by_dirs, dirstate.lt_by_dirs)

    def test__read_dirblocks(self):
        if compiled_dirstate_helpers_feature.available():
            from .._dirstate_helpers_pyx import _read_dirblocks
        else:
            from .._dirstate_helpers_py import _read_dirblocks
        self.assertIs(_read_dirblocks, dirstate._read_dirblocks)

    def test_update_entry(self):
        if compiled_dirstate_helpers_feature.available():
            from .._dirstate_helpers_pyx import update_entry
        else:
            from ..dirstate import update_entry
        self.assertIs(update_entry, dirstate.update_entry)

    def test_process_entry(self):
        if compiled_dirstate_helpers_feature.available():
            from .._dirstate_helpers_pyx import ProcessEntryC
            self.assertIs(ProcessEntryC, dirstate._process_entry)
        else:
            from ..dirstate import ProcessEntryPython
            self.assertIs(ProcessEntryPython, dirstate._process_entry)