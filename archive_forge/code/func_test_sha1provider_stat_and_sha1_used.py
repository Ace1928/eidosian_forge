import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def test_sha1provider_stat_and_sha1_used(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/file'])
    tree.add(['file'], ids=[b'file-id'])
    tree.commit('one')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    state = tree._current_dirstate()
    state._sha1_provider = UppercaseSHA1Provider()
    self.assertChangedFileIds([b'file-id'], tree)