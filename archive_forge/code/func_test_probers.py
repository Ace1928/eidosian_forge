from typing import List
from .. import urlutils
from ..branch import Branch
from ..bzr import BzrProber
from ..bzr.branch import BranchReferenceFormat
from ..controldir import ControlDir, ControlDirFormat
from ..errors import NotBranchError, RedirectRequested
from ..transport import (Transport, chroot, get_transport, register_transport,
from ..url_policy_open import (BadUrl, BranchLoopError, BranchOpener,
from . import TestCase, TestCaseWithTransport
def test_probers(self):
    b = self.make_branch('branch')
    opener = self.make_branch_opener([b.base], probers=[])
    self.assertRaises(NotBranchError, opener.open, b.base)
    opener = self.make_branch_opener([b.base], probers=[BzrProber])
    self.assertEqual(b.base, opener.open(b.base).base)