from unittest import mock
from neutron_lib.callbacks import events
from neutron_lib.callbacks import resources
from neutron_lib import fixture
from neutron_lib import worker
from neutron_lib.tests import _base as base
def test_worker_process_count(self):
    self.assertEqual(9, _BaseWorker(worker_process_count=9).worker_process_count)