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
class TestRepositoryGetParentMap(TestRemoteRepository):

    def test_get_parent_map_caching(self):
        r1 = 'ำ'.encode()
        r2 = 'ණ'.encode()
        lines = [b' '.join([r2, r1]), r1]
        encoded_body = bz2.compress(b'\n'.join(lines))
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_success_response_with_body(encoded_body, b'ok')
        client.add_success_response_with_body(encoded_body, b'ok')
        repo.lock_read()
        graph = repo.get_graph()
        parents = graph.get_parent_map([r2])
        self.assertEqual({r2: (r1,)}, parents)
        repo.lock_read()
        repo.unlock()
        parents = graph.get_parent_map([r1])
        self.assertEqual({r1: (NULL_REVISION,)}, parents)
        self.assertEqual([('call_with_body_bytes_expecting_body', b'Repository.get_parent_map', (b'quack/', b'include-missing:', r2), b'\n\n0')], client._calls)
        repo.unlock()
        repo.lock_read()
        graph = repo.get_graph()
        parents = graph.get_parent_map([r1])
        self.assertEqual({r1: (NULL_REVISION,)}, parents)
        self.assertEqual([('call_with_body_bytes_expecting_body', b'Repository.get_parent_map', (b'quack/', b'include-missing:', r2), b'\n\n0'), ('call_with_body_bytes_expecting_body', b'Repository.get_parent_map', (b'quack/', b'include-missing:', r1), b'\n\n0')], client._calls)
        repo.unlock()

    def test_get_parent_map_reconnects_if_unknown_method(self):
        transport_path = 'quack'
        rev_id = b'revision-id'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_unknown_method_response(b'Repository.get_parent_map')
        client.add_success_response_with_body(rev_id, b'ok')
        self.assertFalse(client._medium._is_remote_before((1, 2)))
        parents = repo.get_parent_map([rev_id])
        self.assertEqual([('call_with_body_bytes_expecting_body', b'Repository.get_parent_map', (b'quack/', b'include-missing:', rev_id), b'\n\n0'), ('disconnect medium',), ('call_expecting_body', b'Repository.get_revision_graph', (b'quack/', b''))], client._calls)
        self.assertTrue(client._medium._is_remote_before((1, 2)))
        self.assertEqual({rev_id: (b'null:',)}, parents)

    def test_get_parent_map_fallback_parentless_node(self):
        """get_parent_map falls back to get_revision_graph on old servers.  The
        results from get_revision_graph are tweaked to match the get_parent_map
        API.

        Specifically, a {key: ()} result from get_revision_graph means "no
        parents" for that key, which in get_parent_map results should be
        represented as {key: ('null:',)}.

        This is the test for https://bugs.launchpad.net/bzr/+bug/214894
        """
        rev_id = b'revision-id'
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_success_response_with_body(rev_id, b'ok')
        client._medium._remember_remote_is_before((1, 2))
        parents = repo.get_parent_map([rev_id])
        self.assertEqual([('call_expecting_body', b'Repository.get_revision_graph', (b'quack/', b''))], client._calls)
        self.assertEqual({rev_id: (b'null:',)}, parents)

    def test_get_parent_map_unexpected_response(self):
        repo, client = self.setup_fake_client_and_repository('path')
        client.add_success_response(b'something unexpected!')
        self.assertRaises(errors.UnexpectedSmartServerResponse, repo.get_parent_map, [b'a-revision-id'])

    def test_get_parent_map_negative_caches_missing_keys(self):
        self.setup_smart_server_with_call_log()
        repo = self.make_repository('foo')
        self.assertIsInstance(repo, RemoteRepository)
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.reset_smart_call_log()
        graph = repo.get_graph()
        self.assertEqual({}, graph.get_parent_map([b'some-missing', b'other-missing']))
        self.assertLength(1, self.hpss_calls)
        self.reset_smart_call_log()
        graph = repo.get_graph()
        self.assertEqual({}, graph.get_parent_map([b'some-missing', b'other-missing']))
        self.assertLength(0, self.hpss_calls)
        self.reset_smart_call_log()
        graph = repo.get_graph()
        self.assertEqual({}, graph.get_parent_map([b'some-missing', b'other-missing', b'more-missing']))
        self.assertLength(1, self.hpss_calls)

    def disableExtraResults(self):
        self.overrideAttr(SmartServerRepositoryGetParentMap, 'no_extra_results', True)

    def test_null_cached_missing_and_stop_key(self):
        self.setup_smart_server_with_call_log()
        builder = self.make_branch_builder('foo')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', ''))], revision_id=b'first')
        builder.finish_series()
        branch = builder.get_branch()
        repo = branch.repository
        self.assertIsInstance(repo, RemoteRepository)
        self.disableExtraResults()
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.reset_smart_call_log()
        graph = repo.get_graph()
        self.assertEqual({b'first': (b'null:',)}, graph.get_parent_map([b'first', b'null:']))
        self.assertEqual({}, graph.get_parent_map([b'another-key']))
        self.assertLength(2, self.hpss_calls)

    def test_get_parent_map_gets_ghosts_from_result(self):
        self.setup_smart_server_with_call_log()
        tree = self.make_branch_and_memory_tree('foo')
        with tree.lock_write():
            builder = treebuilder.TreeBuilder()
            builder.start_tree(tree)
            builder.build([])
            builder.finish_tree()
            tree.set_parent_ids([b'non-existant'], allow_leftmost_as_ghost=True)
            rev_id = tree.commit('')
        tree.lock_read()
        self.addCleanup(tree.unlock)
        repo = tree.branch.repository
        self.assertIsInstance(repo, RemoteRepository)
        repo.get_parent_map([rev_id])
        self.reset_smart_call_log()
        self.assertEqual({}, repo.get_parent_map([b'non-existant']))
        self.assertLength(0, self.hpss_calls)

    def test_exposes_get_cached_parent_map(self):
        """RemoteRepository exposes get_cached_parent_map from
        _unstacked_provider
        """
        r1 = 'ำ'.encode()
        r2 = 'ණ'.encode()
        lines = [b' '.join([r2, r1]), r1]
        encoded_body = bz2.compress(b'\n'.join(lines))
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_success_response_with_body(encoded_body, b'ok')
        repo.lock_read()
        self.assertEqual({}, repo.get_cached_parent_map([r1]))
        self.assertEqual([], client._calls)
        self.assertEqual({r2: (r1,)}, repo.get_parent_map([r2]))
        self.assertEqual({r1: (NULL_REVISION,)}, repo.get_cached_parent_map([r1]))
        self.assertEqual([('call_with_body_bytes_expecting_body', b'Repository.get_parent_map', (b'quack/', b'include-missing:', r2), b'\n\n0')], client._calls)
        repo.unlock()