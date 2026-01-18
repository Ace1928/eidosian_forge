import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_upload_noauto(self):
    """Test that upload --no-auto unsets the upload_auto option"""
    self.make_branch_and_working_tree()
    self.add_file('hello', b'foo')
    self.do_full_upload(auto=True)
    self.assertUpFileEqual(b'foo', 'hello')
    self.assertTrue(self.get_upload_auto())
    self.add_file('bye', b'bar')
    self.do_full_upload(auto=False)
    self.assertUpFileEqual(b'bar', 'bye')
    self.assertFalse(self.get_upload_auto())
    self.add_file('again', b'baz')
    self.do_full_upload()
    self.assertUpFileEqual(b'baz', 'again')
    self.assertFalse(self.get_upload_auto())