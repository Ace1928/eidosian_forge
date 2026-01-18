import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_tricky_paths(self):
    self.assertLtPathByDirblock([b'', b'a', b'a-a', b'a=a', b'b', b'a/a', b'a/a-a', b'a/a=a', b'a/b', b'a/a/a', b'a/a/a-a', b'a/a/a=a', b'a/a/a/a', b'a/a/a/b', b'a/a/a-a/a', b'a/a/a-a/b', b'a/a/a=a/a', b'a/a/a=a/b', b'a/a-a/a', b'a/a-a/a/a', b'a/a-a/a/b', b'a/a=a/a', b'a/b/a', b'a/b/b', b'a-a/a', b'a-a/b', b'a=a/a', b'a=a/b', b'b/a', b'b/b'])
    self.assertLtPathByDirblock([b'', b'a', b'a-a', b'a-z', b'a=a', b'a=z', b'a/a', b'a/a-a', b'a/a-z', b'a/a=a', b'a/a=z', b'a/z', b'a/z-a', b'a/z-z', b'a/z=a', b'a/z=z', b'a/a/a', b'a/a/z', b'a/a-a/a', b'a/a-z/z', b'a/a=a/a', b'a/a=z/z', b'a/z/a', b'a/z/z', b'a-a/a', b'a-z/z', b'a=a/a', b'a=z/z'])