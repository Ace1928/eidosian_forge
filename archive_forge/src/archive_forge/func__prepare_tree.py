import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
def _prepare_tree(self):
    text = b'Hello World\n'
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/a file', text)])
    tree.add('a file', ids=b'a-file-id')
    tree.commit('first')
    return (tree, text)