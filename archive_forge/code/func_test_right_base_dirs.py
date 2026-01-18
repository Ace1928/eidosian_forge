import os
import subprocess
import sys
import breezy.branch
import breezy.bzr.branch
from ... import (branch, bzr, config, controldir, errors, help_topics, lock,
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ... import urlutils, win32utils
from ...errors import (NotBranchError, UnknownFormatError,
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from ...transport import memory, pathfilter
from ...transport.http.urllib import HttpTransport
from ...transport.nosmart import NoSmartTransportDecorator
from ...transport.readonly import ReadonlyTransportDecorator
from .. import branch as bzrbranch
from .. import (bzrdir, knitpack_repo, knitrepo, remote, workingtree_3,
from ..fullhistory import BzrBranchFormat5
def test_right_base_dirs(self):
    dir = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
    t = dir.transport
    branch_base = t.clone('branch').base
    self.assertEqual(branch_base, dir.get_branch_transport(None).base)
    self.assertEqual(branch_base, dir.get_branch_transport(BzrBranchFormat5()).base)
    repository_base = t.clone('repository').base
    self.assertEqual(repository_base, dir.get_repository_transport(None).base)
    repository_format = repository.format_registry.get_default()
    self.assertEqual(repository_base, dir.get_repository_transport(repository_format).base)
    checkout_base = t.clone('checkout').base
    self.assertEqual(checkout_base, dir.get_workingtree_transport(None).base)
    self.assertEqual(checkout_base, dir.get_workingtree_transport(workingtree_3.WorkingTreeFormat3()).base)