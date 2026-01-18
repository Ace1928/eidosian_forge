import errno
from stat import S_ISDIR
import breezy.branch
from breezy import controldir, errors, repository
from breezy import revision as _mod_revision
from breezy import transport, workingtree
from breezy.bzr import bzrdir
from breezy.bzr.remote import RemoteBzrDirFormat
from breezy.bzr.tests.per_bzrdir import TestCaseWithBzrDir
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.transport import FileExists
from breezy.transport.local import LocalTransport
def test_retire_bzrdir(self):
    bd = self.make_controldir('.')
    transport = bd.root_transport
    self.build_tree(['.bzr.retired.0/', '.bzr.retired.0/junk'], transport=transport)
    self.assertTrue(transport.has('.bzr'))
    bd.retire_bzrdir()
    self.assertFalse(transport.has('.bzr'))
    self.assertTrue(transport.has('.bzr.retired.1'))