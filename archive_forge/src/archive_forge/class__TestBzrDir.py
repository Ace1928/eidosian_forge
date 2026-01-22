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
class _TestBzrDir(bzrdir.BzrDirMeta1):
    """Test BzrDir implementation for TestBzrDirSprout.

    When created a _TestBzrDir already has repository and a branch.  The branch
    is a test double as well.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_branch = _TestBranch(self.transport)
        self.test_branch.repository = self.create_repository()

    def open_branch(self, unsupported=False, possible_transports=None):
        return self.test_branch

    def cloning_metadir(self, require_stacking=False):
        return _TestBzrDirFormat()