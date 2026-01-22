from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@mock.patch.object(node.Node, '_assert_microversion_for', _fake_assert)
@mock.patch.object(node.Node, 'fetch', lambda self, session: self)
@mock.patch.object(exceptions, 'raise_from_response', mock.Mock())
class TestNodeSetProvisionState(base.TestCase):

    def setUp(self):
        super(TestNodeSetProvisionState, self).setUp()
        self.node = node.Node(**FAKE)
        self.session = mock.Mock(spec=adapter.Adapter, default_microversion=None)

    def test_no_arguments(self):
        result = self.node.set_provision_state(self.session, 'active')
        self.assertIs(result, self.node)
        self.session.put.assert_called_once_with('nodes/%s/states/provision' % self.node.id, json={'target': 'active'}, headers=mock.ANY, microversion=None, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_manage(self):
        result = self.node.set_provision_state(self.session, 'manage')
        self.assertIs(result, self.node)
        self.session.put.assert_called_once_with('nodes/%s/states/provision' % self.node.id, json={'target': 'manage'}, headers=mock.ANY, microversion='1.4', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_deploy_with_configdrive(self):
        result = self.node.set_provision_state(self.session, 'active', config_drive='abcd')
        self.assertIs(result, self.node)
        self.session.put.assert_called_once_with('nodes/%s/states/provision' % self.node.id, json={'target': 'active', 'configdrive': 'abcd'}, headers=mock.ANY, microversion=None, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_rebuild_with_configdrive(self):
        result = self.node.set_provision_state(self.session, 'rebuild', config_drive='abcd')
        self.assertIs(result, self.node)
        self.session.put.assert_called_once_with('nodes/%s/states/provision' % self.node.id, json={'target': 'rebuild', 'configdrive': 'abcd'}, headers=mock.ANY, microversion='1.35', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_configdrive_as_dict(self):
        for target in ('rebuild', 'active'):
            self.session.put.reset_mock()
            result = self.node.set_provision_state(self.session, target, config_drive={'user_data': 'abcd'})
            self.assertIs(result, self.node)
            self.session.put.assert_called_once_with('nodes/%s/states/provision' % self.node.id, json={'target': target, 'configdrive': {'user_data': 'abcd'}}, headers=mock.ANY, microversion='1.56', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_deploy_with_deploy_steps(self):
        deploy_steps = [{'interface': 'deploy', 'step': 'upgrade_fw'}]
        result = self.node.set_provision_state(self.session, 'active', deploy_steps=deploy_steps)
        self.assertIs(result, self.node)
        self.session.put.assert_called_once_with('nodes/%s/states/provision' % self.node.id, json={'target': 'active', 'deploy_steps': deploy_steps}, headers=mock.ANY, microversion='1.69', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_rebuild_with_deploy_steps(self):
        deploy_steps = [{'interface': 'deploy', 'step': 'upgrade_fw'}]
        result = self.node.set_provision_state(self.session, 'rebuild', deploy_steps=deploy_steps)
        self.assertIs(result, self.node)
        self.session.put.assert_called_once_with('nodes/%s/states/provision' % self.node.id, json={'target': 'rebuild', 'deploy_steps': deploy_steps}, headers=mock.ANY, microversion='1.69', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_set_provision_state_unhold(self):
        result = self.node.set_provision_state(self.session, 'unhold')
        self.assertIs(result, self.node)
        self.session.put.assert_called_once_with('nodes/%s/states/provision' % self.node.id, json={'target': 'unhold'}, headers=mock.ANY, microversion='1.85', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_set_provision_state_service(self):
        service_steps = [{'interface': 'deploy', 'step': 'hold'}]
        result = self.node.set_provision_state(self.session, 'service', service_steps=service_steps)
        self.assertIs(result, self.node)
        self.session.put.assert_called_once_with('nodes/%s/states/provision' % self.node.id, json={'target': 'service', 'service_steps': service_steps}, headers=mock.ANY, microversion='1.87', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)