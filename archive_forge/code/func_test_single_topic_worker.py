from oslo_utils import reflection
from taskflow.engines.worker_based import types as worker_types
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
def test_single_topic_worker(self):
    finder = worker_types.ProxyWorkerFinder('me', mock.MagicMock(), [])
    w, emit = finder._add('dummy-topic', [utils.DummyTask])
    self.assertIsNotNone(w)
    self.assertTrue(emit)
    self.assertEqual(1, finder.total_workers)
    w2 = finder.get_worker_for_task(utils.DummyTask)
    self.assertEqual(w.identity, w2.identity)