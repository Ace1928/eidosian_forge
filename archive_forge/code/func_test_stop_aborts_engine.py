import collections
import contextlib
import threading
import futurist
import testscenarios
from zake import fake_client
from taskflow.conductors import backends
from taskflow import engines
from taskflow.jobs.backends import impl_zookeeper
from taskflow.jobs import base
from taskflow.patterns import linear_flow as lf
from taskflow.persistence.backends import impl_memory
from taskflow import states as st
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as pu
from taskflow.utils import threading_utils
def test_stop_aborts_engine(self):
    components = self.make_components()
    components.conductor.connect()
    consumed_event = threading.Event()
    job_consumed_event = threading.Event()
    job_abandoned_event = threading.Event()
    running_start_event = threading.Event()

    def on_running_start(event, details):
        running_start_event.set()

    def on_consume(state, details):
        consumed_event.set()

    def on_job_consumed(event, details):
        if event == 'job_consumed':
            job_consumed_event.set()

    def on_job_abandoned(event, details):
        if event == 'job_abandoned':
            job_abandoned_event.set()
    components.board.notifier.register(base.REMOVAL, on_consume)
    components.conductor.notifier.register('job_consumed', on_job_consumed)
    components.conductor.notifier.register('job_abandoned', on_job_abandoned)
    components.conductor.notifier.register('running_start', on_running_start)
    with close_many(components.conductor, components.client):
        t = threading_utils.daemon_thread(components.conductor.run)
        t.start()
        lb, fd = pu.temporary_flow_detail(components.persistence)
        engines.save_factory_details(fd, sleep_factory, [], {}, backend=components.persistence)
        components.board.post('poke', lb, details={'flow_uuid': fd.uuid, 'store': {'duration': 2}})
        running_start_event.wait(test_utils.WAIT_TIMEOUT)
        components.conductor.stop()
        job_abandoned_event.wait(test_utils.WAIT_TIMEOUT)
        self.assertTrue(job_abandoned_event.is_set())
        self.assertFalse(job_consumed_event.is_set())
        self.assertFalse(consumed_event.is_set())