import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
class TestBisectPathRight(tests.TestCase, TestBisectPathMixin):
    """Run all Bisect Path tests against _bisect_path_right"""

    def get_bisect_path(self):
        from .._dirstate_helpers_py import _bisect_path_right
        return _bisect_path_right

    def get_bisect(self):
        return (bisect.bisect_right, -1)