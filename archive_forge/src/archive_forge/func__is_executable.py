import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def _is_executable(self, mode, old_executable):
    self._log.append(('is_exec', mode, old_executable))
    return super()._is_executable(mode, old_executable)