import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def adjust_time(self, secs):
    """Move the clock forward or back.

        :param secs: The amount to adjust the clock by. Positive values make it
        seem as if we are in the future, negative values make it seem like we
        are in the past.
        """
    self._time_offset += secs
    self._cutoff_time = None