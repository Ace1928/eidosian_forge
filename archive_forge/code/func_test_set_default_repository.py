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
def test_set_default_repository(self):
    default_factory = controldir.format_registry.get('default')
    old_default = [k for k, v in controldir.format_registry.iteritems() if v == default_factory and k != 'default'][0]
    controldir.format_registry.set_default_repository('dirstate-with-subtree')
    try:
        self.assertIs(controldir.format_registry.get('dirstate-with-subtree'), controldir.format_registry.get('default'))
        self.assertIs(repository.format_registry.get_default().__class__, knitrepo.RepositoryFormatKnit3)
    finally:
        controldir.format_registry.set_default_repository(old_default)