import contextlib
import threading
from kazoo.protocol import paths as k_paths
from kazoo.recipe import watchers
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
import testtools
from zake import fake_client
from zake import utils as zake_utils
from taskflow import exceptions as excp
from taskflow.jobs.backends import impl_zookeeper
from taskflow import states
from taskflow import test
from taskflow.test import mock
from taskflow.tests.unit.jobs import base
from taskflow.tests import utils as test_utils
from taskflow.types import entity
from taskflow.utils import kazoo_utils
from taskflow.utils import misc
from taskflow.utils import persistence_utils as p_utils
class ZakeJobboardTest(test.TestCase, ZookeeperBoardTestMixin):

    def create_board(self, persistence=None):
        client = fake_client.FakeClient()
        board = impl_zookeeper.ZookeeperJobBoard('test-board', {}, client=client, persistence=persistence)
        self.addCleanup(board.close)
        self.addCleanup(self.close_client, client)
        return (client, board)

    def setUp(self):
        super(ZakeJobboardTest, self).setUp()
        self.client, self.board = self.create_board()
        self.bad_paths = [self.board.path, self.board.trash_path]
        self.bad_paths.extend(zake_utils.partition_path(self.board.path))

    def test_posting_owner_lost(self):
        with base.connect_close(self.board):
            with self.flush(self.client):
                j = self.board.post('test', p_utils.temporary_log_book())
            self.assertEqual(states.UNCLAIMED, j.state)
            with self.flush(self.client):
                self.board.claim(j, self.board.name)
            self.assertEqual(states.CLAIMED, j.state)
            paths = list(self.client.storage.paths.items())
            for path, value in paths:
                if path in self.bad_paths:
                    continue
                if path.endswith('lock'):
                    value['data'] = misc.binary_encode(jsonutils.dumps({}))
            self.assertEqual(states.UNCLAIMED, j.state)

    def test_posting_state_lock_lost(self):
        with base.connect_close(self.board):
            with self.flush(self.client):
                j = self.board.post('test', p_utils.temporary_log_book())
            self.assertEqual(states.UNCLAIMED, j.state)
            with self.flush(self.client):
                self.board.claim(j, self.board.name)
            self.assertEqual(states.CLAIMED, j.state)
            paths = list(self.client.storage.paths.items())
            for path, value in paths:
                if path in self.bad_paths:
                    continue
                if path.endswith('lock'):
                    self.client.storage.pop(path)
            self.assertEqual(states.UNCLAIMED, j.state)

    def test_trashing_claimed_job(self):
        with base.connect_close(self.board):
            with self.flush(self.client):
                j = self.board.post('test', p_utils.temporary_log_book())
            self.assertEqual(states.UNCLAIMED, j.state)
            with self.flush(self.client):
                self.board.claim(j, self.board.name)
            self.assertEqual(states.CLAIMED, j.state)
            with self.flush(self.client):
                self.board.trash(j, self.board.name)
            trashed = []
            jobs = []
            paths = list(self.client.storage.paths.items())
            for path, value in paths:
                if path in self.bad_paths:
                    continue
                if path.find(TRASH_FOLDER) > -1:
                    trashed.append(path)
                elif path.find(self.board._job_base) > -1 and (not path.endswith(LOCK_POSTFIX)):
                    jobs.append(path)
            self.assertEqual(1, len(trashed))
            self.assertEqual(0, len(jobs))

    def test_posting_received_raw(self):
        book = p_utils.temporary_log_book()
        with base.connect_close(self.board):
            self.assertTrue(self.board.connected)
            self.assertEqual(0, self.board.job_count)
            posted_job = self.board.post('test', book)
            self.assertEqual(self.board, posted_job.board)
            self.assertEqual(1, self.board.job_count)
            self.assertIn(posted_job.uuid, [j.uuid for j in self.board.iterjobs()])
        paths = {}
        for path, data in self.client.storage.paths.items():
            if path in self.bad_paths:
                continue
            paths[path] = data
        self.assertEqual(1, len(paths))
        path_key = list(paths.keys())[0]
        self.assertTrue(len(paths[path_key]['data']) > 0)
        self.assertDictEqual({'uuid': posted_job.uuid, 'name': posted_job.name, 'book': {'name': book.name, 'uuid': book.uuid}, 'priority': 'NORMAL', 'details': {}}, jsonutils.loads(misc.binary_decode(paths[path_key]['data'])))

    def test_register_entity(self):
        conductor_name = 'conductor-abc@localhost:4123'
        entity_instance = entity.Entity('conductor', conductor_name, {})
        with base.connect_close(self.board):
            self.board.register_entity(entity_instance)
        self.assertIn(self.board.entity_path, self.client.storage.paths)
        conductor_entity_path = k_paths.join(self.board.entity_path, 'conductor', conductor_name)
        self.assertIn(conductor_entity_path, self.client.storage.paths)
        conductor_data = self.client.storage.paths[conductor_entity_path]['data']
        self.assertTrue(len(conductor_data) > 0)
        self.assertDictEqual({'name': conductor_name, 'kind': 'conductor', 'metadata': {}}, jsonutils.loads(misc.binary_decode(conductor_data)))
        entity_instance_2 = entity.Entity('non-sense', 'other_name', {})
        with base.connect_close(self.board):
            self.assertRaises(excp.NotImplementedError, self.board.register_entity, entity_instance_2)

    def test_connect_check_compatible(self):
        client = fake_client.FakeClient()
        board = impl_zookeeper.ZookeeperJobBoard('test-board', {'check_compatible': True}, client=client)
        self.addCleanup(board.close)
        self.addCleanup(self.close_client, client)
        with base.connect_close(board):
            pass
        client = fake_client.FakeClient(server_version=(3, 2, 0))
        board = impl_zookeeper.ZookeeperJobBoard('test-board', {'check_compatible': False}, client=client)
        self.addCleanup(board.close)
        self.addCleanup(self.close_client, client)
        with base.connect_close(board):
            pass
        client = fake_client.FakeClient(server_version=(3, 2, 0))
        board = impl_zookeeper.ZookeeperJobBoard('test-board', {'check_compatible': True}, client=client)
        self.addCleanup(board.close)
        self.addCleanup(self.close_client, client)
        self.assertRaises(excp.IncompatibleVersion, board.connect)
        client = fake_client.FakeClient(server_version=(3, 2, 0))
        board = impl_zookeeper.ZookeeperJobBoard('test-board', {'check_compatible': 'False'}, client=client)
        self.addCleanup(board.close)
        self.addCleanup(self.close_client, client)
        with base.connect_close(board):
            pass