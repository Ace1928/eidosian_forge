import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def assertCmpByDirs(self, expected, str1, str2):
    """Compare the two strings, in both directions.

        :param expected: The expected comparison value. -1 means str1 comes
            first, 0 means they are equal, 1 means str2 comes first
        :param str1: string to compare
        :param str2: string to compare
        """
    lt_by_dirs = self.get_lt_by_dirs()
    if expected == 0:
        self.assertEqual(str1, str2)
        self.assertFalse(lt_by_dirs(str1, str2))
        self.assertFalse(lt_by_dirs(str2, str1))
    elif expected > 0:
        self.assertFalse(lt_by_dirs(str1, str2))
        self.assertTrue(lt_by_dirs(str2, str1))
    else:
        self.assertTrue(lt_by_dirs(str1, str2))
        self.assertFalse(lt_by_dirs(str2, str1))