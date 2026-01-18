import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_pack_stat_float(self):
    """On some platforms mtime and ctime are floats.

        Make sure we don't get warnings or errors, and that we ignore changes <
        1s
        """
    st = _FakeStat(7000, 1172758614.0, 1172758617.0, 777, 6499538, 33188)
    self.assertPackStat(b'AAAbWEXm4FZF5uBZAAADCQBjLNIAAIGk', st)
    st.st_mtime = 1172758620.0
    self.assertPackStat(b'AAAbWEXm4FxF5uBZAAADCQBjLNIAAIGk', st)
    st.st_ctime = 1172758630.0
    self.assertPackStat(b'AAAbWEXm4FxF5uBmAAADCQBjLNIAAIGk', st)
    st.st_mtime = 1172758620.453
    self.assertPackStat(b'AAAbWEXm4FxF5uBmAAADCQBjLNIAAIGk', st)
    st.st_ctime = 1172758630.228
    self.assertPackStat(b'AAAbWEXm4FxF5uBmAAADCQBjLNIAAIGk', st)