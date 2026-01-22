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
class TestRemoteBranchEffort(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.smart_server = test_server.SmartTCPServer_for_testing()
        self.start_server(self.smart_server, self.get_server())
        _SmartClient.hooks.install_named_hook('call', self.capture_hpss_call, None)
        self.hpss_calls = []

    def capture_hpss_call(self, params):
        self.hpss_calls.append(params.method)

    def test_copy_content_into_avoids_revision_history(self):
        local = self.make_branch('local')
        builder = self.make_branch_builder('remote')
        builder.build_commit(message='Commit.')
        remote_branch_url = self.smart_server.get_url() + 'remote'
        remote_branch = bzrdir.BzrDir.open(remote_branch_url).open_branch()
        local.repository.fetch(remote_branch.repository)
        self.hpss_calls = []
        remote_branch.copy_content_into(local)
        self.assertFalse(b'Branch.revision_history' in self.hpss_calls)

    def test_fetch_everything_needs_just_one_call(self):
        local = self.make_branch('local')
        builder = self.make_branch_builder('remote')
        builder.build_commit(message='Commit.')
        remote_branch_url = self.smart_server.get_url() + 'remote'
        remote_branch = bzrdir.BzrDir.open(remote_branch_url).open_branch()
        self.hpss_calls = []
        local.repository.fetch(remote_branch.repository, fetch_spec=vf_search.EverythingResult(remote_branch.repository))
        self.assertEqual([b'Repository.get_stream_1.19'], self.hpss_calls)

    def override_verb(self, verb_name, verb):
        request_handlers = request.request_handlers
        orig_verb = request_handlers.get(verb_name)
        orig_info = request_handlers.get_info(verb_name)
        request_handlers.register(verb_name, verb, override_existing=True)
        self.addCleanup(request_handlers.register, verb_name, orig_verb, override_existing=True, info=orig_info)

    def test_fetch_everything_backwards_compat(self):
        """Can fetch with EverythingResult even with pre 2.4 servers.

        Pre-2.4 do not support 'everything' searches with the
        Repository.get_stream_1.19 verb.
        """
        verb_log = []

        class OldGetStreamVerb(SmartServerRepositoryGetStream_1_19):
            """A version of the Repository.get_stream_1.19 verb patched to
            reject 'everything' searches the way 2.3 and earlier do.
            """

            def recreate_search(self, repository, search_bytes, discard_excess=False):
                verb_log.append(search_bytes.split(b'\n', 1)[0])
                if search_bytes == b'everything':
                    return (None, request.FailedSmartServerResponse((b'BadSearch',)))
                return super().recreate_search(repository, search_bytes, discard_excess=discard_excess)
        self.override_verb(b'Repository.get_stream_1.19', OldGetStreamVerb)
        local = self.make_branch('local')
        builder = self.make_branch_builder('remote')
        builder.build_commit(message='Commit.')
        remote_branch_url = self.smart_server.get_url() + 'remote'
        remote_branch = bzrdir.BzrDir.open(remote_branch_url).open_branch()
        self.hpss_calls = []
        local.repository.fetch(remote_branch.repository, fetch_spec=vf_search.EverythingResult(remote_branch.repository))
        self.assertLength(1, verb_log)
        self.assertTrue(len(self.hpss_calls) > 1)