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
def test_stacked_outside_scheme(self):
    self.get_transport().mkdir('inside')
    self.get_transport().mkdir('outside')
    self.make_branch('inside/stacked')
    self.make_branch('outside/stacked-on')
    scheme, get_chrooted_url = self.get_chrooted_scheme('inside')
    Branch.open(get_chrooted_url('stacked')).set_stacked_on_url(self.get_url('outside/stacked-on'))
    self.assertRaises(BadUrl, open_only_scheme, scheme, get_chrooted_url('stacked'))