import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_wait_for_baremetal_node_lock_not_locked(self):
    self.fake_baremetal_node['reservation'] = None
    self.assertIsNone(self.cloud.wait_for_baremetal_node_lock(self.fake_baremetal_node, timeout=1))
    self.assertEqual(3, len(self.adapter.request_history))