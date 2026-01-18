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
def test_open_from_transport(self):
    control = bzrdir.BzrDir.create(self.get_url())
    t = self.get_transport()
    opened_bzrdir = bzrdir.BzrDir.open_from_transport(t)
    self.assertEqual(t.base, opened_bzrdir.root_transport.base)
    self.assertIsInstance(opened_bzrdir, bzrdir.BzrDir)