import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_pack_stat_int(self):
    st = _FakeStat(6859, 1172758614, 1172758617, 777, 6499538, 33188)
    self.assertPackStat(b'AAAay0Xm4FZF5uBZAAADCQBjLNIAAIGk', st)
    st.st_size = 7000
    self.assertPackStat(b'AAAbWEXm4FZF5uBZAAADCQBjLNIAAIGk', st)
    st.st_mtime = 1172758620
    self.assertPackStat(b'AAAbWEXm4FxF5uBZAAADCQBjLNIAAIGk', st)
    st.st_ctime = 1172758630
    self.assertPackStat(b'AAAbWEXm4FxF5uBmAAADCQBjLNIAAIGk', st)
    st.st_dev = 888
    self.assertPackStat(b'AAAbWEXm4FxF5uBmAAADeABjLNIAAIGk', st)
    st.st_ino = 6499540
    self.assertPackStat(b'AAAbWEXm4FxF5uBmAAADeABjLNQAAIGk', st)
    st.st_mode = 33252
    self.assertPackStat(b'AAAbWEXm4FxF5uBmAAADeABjLNQAAIHk', st)