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
def test_run_empty(self):
    components = self.make_components()
    components.conductor.connect()
    with close_many(components.conductor, components.client):
        t = threading_utils.daemon_thread(components.conductor.run)
        t.start()
        components.conductor.stop()
        self.assertTrue(components.conductor.wait(test_utils.WAIT_TIMEOUT))
        self.assertFalse(components.conductor.dispatching)
        t.join()