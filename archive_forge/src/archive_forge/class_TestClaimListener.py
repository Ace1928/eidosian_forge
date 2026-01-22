import contextlib
import logging
import threading
import time
from oslo_serialization import jsonutils
from oslo_utils import reflection
from zake import fake_client
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.jobs import backends as jobs
from taskflow.listeners import claims
from taskflow.listeners import logging as logging_listeners
from taskflow.listeners import timing
from taskflow.patterns import linear_flow as lf
from taskflow.persistence.backends import impl_memory
from taskflow import states
from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.utils import misc
from taskflow.utils import persistence_utils
class TestClaimListener(test.TestCase, EngineMakerMixin):

    def _make_dummy_flow(self, count):
        f = lf.Flow('root')
        for i in range(0, count):
            f.add(test_utils.ProvidesRequiresTask('%s_test' % i, [], []))
        return f

    def setUp(self):
        super(TestClaimListener, self).setUp()
        self.client = fake_client.FakeClient()
        self.addCleanup(self.client.stop)
        self.board = jobs.fetch('test', 'zookeeper', client=self.client)
        self.addCleanup(self.board.close)
        self.board.connect()

    def _post_claim_job(self, job_name, book=None, details=None):
        arrived = threading.Event()

        def set_on_children(children):
            if children:
                arrived.set()
        self.client.ChildrenWatch('/taskflow', set_on_children)
        job = self.board.post('test-1')
        self.assertTrue(arrived.wait(test_utils.WAIT_TIMEOUT))
        arrived.clear()
        self.board.claim(job, self.board.name)
        self.assertTrue(arrived.wait(test_utils.WAIT_TIMEOUT))
        self.assertEqual(states.CLAIMED, job.state)
        return job

    def _destroy_locks(self):
        children = self.client.storage.get_children('/taskflow', only_direct=False)
        removed = 0
        for p, data in children.items():
            if p.endswith('.lock'):
                self.client.storage.pop(p)
                removed += 1
        return removed

    def _change_owner(self, new_owner):
        children = self.client.storage.get_children('/taskflow', only_direct=False)
        altered = 0
        for p, data in children.items():
            if p.endswith('.lock'):
                self.client.set(p, misc.binary_encode(jsonutils.dumps({'owner': new_owner})))
                altered += 1
        return altered

    def test_bad_create(self):
        job = self._post_claim_job('test')
        f = self._make_dummy_flow(10)
        e = self._make_engine(f)
        self.assertRaises(ValueError, claims.CheckingClaimListener, e, job, self.board, self.board.name, on_job_loss=1)

    def test_claim_lost_suspended(self):
        job = self._post_claim_job('test')
        f = self._make_dummy_flow(10)
        e = self._make_engine(f)
        try_destroy = True
        ran_states = []
        with claims.CheckingClaimListener(e, job, self.board, self.board.name):
            for state in e.run_iter():
                ran_states.append(state)
                if state == states.SCHEDULING and try_destroy:
                    try_destroy = bool(self._destroy_locks())
        self.assertEqual(states.SUSPENDED, e.storage.get_flow_state())
        self.assertEqual(1, ran_states.count(states.ANALYZING))
        self.assertEqual(1, ran_states.count(states.SCHEDULING))
        self.assertEqual(1, ran_states.count(states.WAITING))

    def test_claim_lost_custom_handler(self):
        job = self._post_claim_job('test')
        f = self._make_dummy_flow(10)
        e = self._make_engine(f)
        handler = mock.MagicMock()
        ran_states = []
        try_destroy = True
        destroyed_at = -1
        with claims.CheckingClaimListener(e, job, self.board, self.board.name, on_job_loss=handler):
            for i, state in enumerate(e.run_iter()):
                ran_states.append(state)
                if state == states.SCHEDULING and try_destroy:
                    destroyed = bool(self._destroy_locks())
                    if destroyed:
                        destroyed_at = i
                        try_destroy = False
        self.assertTrue(handler.called)
        self.assertEqual(10, ran_states.count(states.SCHEDULING))
        self.assertNotEqual(-1, destroyed_at)
        after_states = ran_states[destroyed_at:]
        self.assertGreater(0, len(after_states))

    def test_claim_lost_new_owner(self):
        job = self._post_claim_job('test')
        f = self._make_dummy_flow(10)
        e = self._make_engine(f)
        change_owner = True
        ran_states = []
        with claims.CheckingClaimListener(e, job, self.board, self.board.name):
            for state in e.run_iter():
                ran_states.append(state)
                if state == states.SCHEDULING and change_owner:
                    change_owner = bool(self._change_owner('test-2'))
        self.assertEqual(states.SUSPENDED, e.storage.get_flow_state())
        self.assertEqual(1, ran_states.count(states.ANALYZING))
        self.assertEqual(1, ran_states.count(states.SCHEDULING))
        self.assertEqual(1, ran_states.count(states.WAITING))