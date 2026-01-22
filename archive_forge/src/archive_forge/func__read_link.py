import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def _read_link(self, abspath, old_link):
    self._log.append(('read_link', abspath, old_link))
    return super()._read_link(abspath, old_link)