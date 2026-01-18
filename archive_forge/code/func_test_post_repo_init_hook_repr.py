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
def test_post_repo_init_hook_repr(self):
    param_reprs = []
    bzrdir.BzrDir.hooks.install_named_hook('post_repo_init', lambda params: param_reprs.append(repr(params)), None)
    self.make_repository('foo')
    self.assertLength(1, param_reprs)
    param_repr = param_reprs[0]
    self.assertStartsWith(param_repr, '<RepoInitHookParams for ')