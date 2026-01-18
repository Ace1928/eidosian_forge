import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_path_containing_carriagereturn_skips(self):
    self.assertFilenameSkipped('a\rb')