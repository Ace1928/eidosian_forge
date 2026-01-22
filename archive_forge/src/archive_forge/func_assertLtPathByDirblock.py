import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def assertLtPathByDirblock(self, paths):
    """Compare all paths and make sure they evaluate to the correct order.

        This does N^2 comparisons. It is assumed that ``paths`` is properly
        sorted list.

        :param paths: a sorted list of paths to compare
        """

    def _key(p):
        dirname, basename = os.path.split(p)
        return (dirname.split(b'/'), basename)
    self.assertEqual(sorted(paths, key=_key), paths)
    lt_path_by_dirblock = self.get_lt_path_by_dirblock()
    for idx1, path1 in enumerate(paths):
        for idx2, path2 in enumerate(paths):
            lt_result = lt_path_by_dirblock(path1, path2)
            self.assertEqual(idx1 < idx2, lt_result, '%s did not state that %r < %r, lt=%s' % (lt_path_by_dirblock.__name__, path1, path2, lt_result))