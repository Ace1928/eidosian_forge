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
def test_supports_relative_reference(self):
    tree = self.make_branch_and_tree('.', format='development-colo')
    target1 = tree.controldir.create_branch(name='target1')
    target2 = tree.controldir.create_branch(name='target2')
    source = tree.controldir.set_branch_reference(target1, name='source')
    self.assertEqual(target1.user_url, tree.controldir.open_branch('source').user_url)
    source.controldir.get_branch_transport(None, 'source').put_bytes('location', b'file:,branch=target2')
    self.assertEqual(target2.user_url, tree.controldir.open_branch('source').user_url)