from unittest import mock
from neutron_lib.callbacks import events
from neutron_lib.callbacks import resources
from neutron_lib import fixture
from neutron_lib import worker
from neutron_lib.tests import _base as base
def test_start_callback_event(self):
    base_worker = _BaseWorker()
    base_worker.start()
    self._reg.publish.assert_called_once_with(resources.PROCESS, events.AFTER_INIT, base_worker.start, payload=mock.ANY)