import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def _read_state_content(self, state):
    """Read the content of the dirstate file.

        On Windows when one process locks a file, you can't even open() the
        file in another process (to read it). So we go directly to
        state._state_file. This should always be the exact disk representation,
        so it is reasonable to do so.
        DirState also always seeks before reading, so it doesn't matter if we
        bump the file pointer.
        """
    state._state_file.seek(0)
    return state._state_file.read()