from breezy import errors
from breezy.bzr.tests.per_repository_vf import (
from breezy.bzr.tests.per_repository_vf.helpers import \
from breezy.tests import TestNotApplicable
from breezy.tests.scenarios import load_tests_apply_scenarios
def tree_callback(self, refs):
    self.callbacks.append(('tree', refs))
    return self.tree_check(refs)