from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@mock.patch.object(exceptions, 'raise_from_response', mock.Mock())
@mock.patch.object(node.Node, '_get_session', lambda self, x: x)
class TestNodeValidate(base.TestCase):

    def setUp(self):
        super(TestNodeValidate, self).setUp()
        self.session = mock.Mock(spec=adapter.Adapter)
        self.session.default_microversion = '1.28'
        self.node = node.Node(**FAKE)

    def test_validate_ok(self):
        self.session.get.return_value.json.return_value = {'boot': {'result': True}, 'console': {'result': False, 'reason': 'Not configured'}, 'deploy': {'result': True}, 'inspect': {'result': None, 'reason': 'Not supported'}, 'power': {'result': True}}
        result = self.node.validate(self.session)
        for iface in ('boot', 'deploy', 'power'):
            self.assertTrue(result[iface].result)
            self.assertFalse(result[iface].reason)
        for iface in ('console', 'inspect'):
            self.assertIsNot(True, result[iface].result)
            self.assertTrue(result[iface].reason)

    def test_validate_failed(self):
        self.session.get.return_value.json.return_value = {'boot': {'result': False}, 'console': {'result': False, 'reason': 'Not configured'}, 'deploy': {'result': False, 'reason': 'No deploy for you'}, 'inspect': {'result': None, 'reason': 'Not supported'}, 'power': {'result': True}}
        self.assertRaisesRegex(exceptions.ValidationException, 'No deploy for you', self.node.validate, self.session)

    def test_validate_no_failure(self):
        self.session.get.return_value.json.return_value = {'boot': {'result': False}, 'console': {'result': False, 'reason': 'Not configured'}, 'deploy': {'result': False, 'reason': 'No deploy for you'}, 'inspect': {'result': None, 'reason': 'Not supported'}, 'power': {'result': True}}
        result = self.node.validate(self.session, required=None)
        self.assertTrue(result['power'].result)
        self.assertFalse(result['power'].reason)
        for iface in ('deploy', 'console', 'inspect'):
            self.assertIsNot(True, result[iface].result)
            self.assertTrue(result[iface].reason)
        self.assertFalse(result['boot'].result)
        self.assertIsNone(result['boot'].reason)