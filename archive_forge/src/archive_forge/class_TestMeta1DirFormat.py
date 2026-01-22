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
class TestMeta1DirFormat(TestCaseWithTransport):
    """Tests specific to the meta1 dir format."""

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

    def test_meta1dir_uses_lockdir(self):
        """Meta1 format uses a LockDir to guard the whole directory, not a file."""
        dir = bzrdir.BzrDirMetaFormat1().initialize(self.get_url())
        t = dir.transport
        self.assertIsDirectory('branch-lock', t)

    def test_comparison(self):
        """Equality and inequality behave properly.

        Metadirs should compare equal iff they have the same repo, branch and
        tree formats.
        """
        mydir = controldir.format_registry.make_controldir('knit')
        self.assertEqual(mydir, mydir)
        self.assertFalse(mydir != mydir)
        otherdir = controldir.format_registry.make_controldir('knit')
        self.assertEqual(otherdir, mydir)
        self.assertFalse(otherdir != mydir)
        otherdir2 = controldir.format_registry.make_controldir('development-subtree')
        self.assertNotEqual(otherdir2, mydir)
        self.assertFalse(otherdir2 == mydir)

    def test_with_features(self):
        tree = self.make_branch_and_tree('tree', format='2a')
        tree.controldir.update_feature_flags({b'bar': b'required'})
        self.assertRaises(bzrdir.MissingFeature, bzrdir.BzrDir.open, 'tree')
        bzrdir.BzrDirMetaFormat1.register_feature(b'bar')
        self.addCleanup(bzrdir.BzrDirMetaFormat1.unregister_feature, b'bar')
        dir = bzrdir.BzrDir.open('tree')
        self.assertEqual(b'required', dir._format.features.get(b'bar'))
        tree.controldir.update_feature_flags({b'bar': None, b'nonexistant': None})
        dir = bzrdir.BzrDir.open('tree')
        self.assertEqual({}, dir._format.features)

    def test_needs_conversion_different_working_tree(self):
        new_format = controldir.format_registry.make_controldir('dirstate')
        tree = self.make_branch_and_tree('tree', format='knit')
        self.assertTrue(tree.controldir.needs_format_conversion(new_format))

    def test_initialize_on_format_uses_smart_transport(self):
        self.setup_smart_server_with_call_log()
        new_format = controldir.format_registry.make_controldir('dirstate')
        transport = self.get_transport('target')
        transport.ensure_base()
        self.reset_smart_call_log()
        instance = new_format.initialize_on_transport(transport)
        self.assertIsInstance(instance, remote.RemoteBzrDir)
        rpc_count = len(self.hpss_calls)
        self.assertEqual(2, rpc_count)