import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_rename_two_files(self):
    self.make_branch_and_working_tree()
    self.add_file('a', b'foo')
    self.add_file('b', b'qux')
    self.do_full_upload()
    self.rename_any('b', 'c')
    self.rename_any('a', 'b')
    self.assertUpFileEqual(b'foo', 'a')
    self.assertUpFileEqual(b'qux', 'b')
    self.do_upload()
    self.assertUpFileEqual(b'foo', 'b')
    self.assertUpFileEqual(b'qux', 'c')