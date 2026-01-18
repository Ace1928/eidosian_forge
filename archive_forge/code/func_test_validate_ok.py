from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_validate_ok(self):
    self.session.get.return_value.json.return_value = {'boot': {'result': True}, 'console': {'result': False, 'reason': 'Not configured'}, 'deploy': {'result': True}, 'inspect': {'result': None, 'reason': 'Not supported'}, 'power': {'result': True}}
    result = self.node.validate(self.session)
    for iface in ('boot', 'deploy', 'power'):
        self.assertTrue(result[iface].result)
        self.assertFalse(result[iface].reason)
    for iface in ('console', 'inspect'):
        self.assertIsNot(True, result[iface].result)
        self.assertTrue(result[iface].reason)