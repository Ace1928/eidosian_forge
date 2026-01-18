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