import base64
import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from ... import branch, config, controldir, errors, repository, tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...branch import Branch
from ...revision import NULL_REVISION, Revision
from ...tests import test_server
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from ...transport.remote import (RemoteSSHTransport, RemoteTCPTransport,
from .. import (RemoteBzrProber, bzrdir, groupcompress_repo, inventory,
from ..bzrdir import BzrDir, BzrDirFormat
from ..chk_serializer import chk_bencode_serializer
from ..remote import (RemoteBranch, RemoteBranchFormat, RemoteBzrDir,
from ..smart import medium, request
from ..smart.client import _SmartClient
from ..smart.repository import (SmartServerRepositoryGetParentMap,
class BasicRemoteObjectTests(tests.TestCaseWithTransport):
    scenarios = [('HPSS-v2', {'transport_server': test_server.SmartTCPServer_for_testing_v2_only}), ('HPSS-v3', {'transport_server': test_server.SmartTCPServer_for_testing})]

    def setUp(self):
        super().setUp()
        self.transport = self.get_transport()
        self.local_wt = BzrDir.create_standalone_workingtree('.')
        self.addCleanup(self.transport.disconnect)

    def test_create_remote_bzrdir(self):
        b = remote.RemoteBzrDir(self.transport, RemoteBzrDirFormat())
        self.assertIsInstance(b, BzrDir)

    def test_open_remote_branch(self):
        b = remote.RemoteBzrDir(self.transport, RemoteBzrDirFormat())
        branch = b.open_branch()
        self.assertIsInstance(branch, Branch)

    def test_remote_repository(self):
        b = BzrDir.open_from_transport(self.transport)
        repo = b.open_repository()
        revid = 'È23123123'.encode()
        self.assertFalse(repo.has_revision(revid))
        self.local_wt.commit(message='test commit', rev_id=revid)
        self.assertTrue(repo.has_revision(revid))

    def test_find_correct_format(self):
        """Should open a RemoteBzrDir over a RemoteTransport"""
        fmt = BzrDirFormat.find_format(self.transport)
        self.assertIn(RemoteBzrProber, controldir.ControlDirFormat._probers)
        self.assertIsInstance(fmt, RemoteBzrDirFormat)

    def test_open_detected_smart_format(self):
        fmt = BzrDirFormat.find_format(self.transport)
        d = fmt.open(self.transport)
        self.assertIsInstance(d, BzrDir)

    def test_remote_branch_repr(self):
        b = BzrDir.open_from_transport(self.transport).open_branch()
        self.assertStartsWith(str(b), 'RemoteBranch(')

    def test_remote_bzrdir_repr(self):
        b = BzrDir.open_from_transport(self.transport)
        self.assertStartsWith(str(b), 'RemoteBzrDir(')

    def test_remote_branch_format_supports_stacking(self):
        t = self.transport
        self.make_branch('unstackable', format='pack-0.92')
        b = BzrDir.open_from_transport(t.clone('unstackable')).open_branch()
        self.assertFalse(b._format.supports_stacking())
        self.make_branch('stackable', format='1.9')
        b = BzrDir.open_from_transport(t.clone('stackable')).open_branch()
        self.assertTrue(b._format.supports_stacking())

    def test_remote_repo_format_supports_external_references(self):
        t = self.transport
        bd = self.make_controldir('unstackable', format='pack-0.92')
        r = bd.create_repository()
        self.assertFalse(r._format.supports_external_lookups)
        r = BzrDir.open_from_transport(t.clone('unstackable')).open_repository()
        self.assertFalse(r._format.supports_external_lookups)
        bd = self.make_controldir('stackable', format='1.9')
        r = bd.create_repository()
        self.assertTrue(r._format.supports_external_lookups)
        r = BzrDir.open_from_transport(t.clone('stackable')).open_repository()
        self.assertTrue(r._format.supports_external_lookups)

    def test_remote_branch_set_append_revisions_only(self):
        branch = self.make_branch('branch', format='1.9')
        branch.set_append_revisions_only(True)
        config = branch.get_config_stack()
        self.assertEqual(True, config.get('append_revisions_only'))
        branch.set_append_revisions_only(False)
        config = branch.get_config_stack()
        self.assertEqual(False, config.get('append_revisions_only'))

    def test_remote_branch_set_append_revisions_only_upgrade_reqd(self):
        branch = self.make_branch('branch', format='knit')
        self.assertRaises(errors.UpgradeRequired, branch.set_append_revisions_only, True)