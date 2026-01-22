from unittest import mock
from neutronclient.common import exceptions
from neutronclient.v2_0 import client as neutronclient
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import firewall
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class FirewallPolicyTest(common.HeatTestCase):

    def setUp(self):
        super(FirewallPolicyTest, self).setUp()
        self.mockclient = mock.Mock(spec=neutronclient.Client)
        self.patchobject(neutronclient, 'Client', return_value=self.mockclient)
        self.patchobject(neutron.NeutronClientPlugin, 'has_extension', return_value=True)

    def create_firewall_policy(self):
        self.mockclient.create_firewall_policy.return_value = {'firewall_policy': {'id': '5678'}}
        snippet = template_format.parse(firewall_policy_template)
        self.stack = utils.parse_stack(snippet)
        self.tmpl = snippet
        resource_defns = self.stack.t.resource_definitions(self.stack)
        return firewall.FirewallPolicy('firewall_policy', resource_defns['firewall_policy'], self.stack)

    def test_create(self):
        rsrc = self.create_firewall_policy()
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_firewall_policy.assert_called_once_with({'firewall_policy': {'name': 'test-firewall-policy', 'shared': True, 'audited': True, 'firewall_rules': ['rule-id-1', 'rule-id-2']}})

    def test_create_failed(self):
        self.mockclient.create_firewall_policy.side_effect = exceptions.NeutronClientException()
        snippet = template_format.parse(firewall_policy_template)
        stack = utils.parse_stack(snippet)
        resource_defns = stack.t.resource_definitions(stack)
        rsrc = firewall.FirewallPolicy('firewall_policy', resource_defns['firewall_policy'], stack)
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
        self.assertEqual('NeutronClientException: resources.firewall_policy: An unknown exception occurred.', str(error))
        self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_firewall_policy.assert_called_once_with({'firewall_policy': {'name': 'test-firewall-policy', 'shared': True, 'audited': True, 'firewall_rules': ['rule-id-1', 'rule-id-2']}})

    def test_delete(self):
        rsrc = self.create_firewall_policy()
        self.mockclient.delete_firewall_policy.return_value = None
        self.mockclient.show_firewall_policy.side_effect = exceptions.NeutronClientException(status_code=404)
        scheduler.TaskRunner(rsrc.create)()
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_firewall_policy.assert_called_once_with({'firewall_policy': {'name': 'test-firewall-policy', 'shared': True, 'audited': True, 'firewall_rules': ['rule-id-1', 'rule-id-2']}})
        self.mockclient.delete_firewall_policy.assert_called_once_with('5678')
        self.mockclient.show_firewall_policy.assert_called_once_with('5678')

    def test_delete_already_gone(self):
        rsrc = self.create_firewall_policy()
        self.mockclient.delete_firewall_policy.side_effect = exceptions.NeutronClientException(status_code=404)
        scheduler.TaskRunner(rsrc.create)()
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_firewall_policy.assert_called_once_with({'firewall_policy': {'name': 'test-firewall-policy', 'shared': True, 'audited': True, 'firewall_rules': ['rule-id-1', 'rule-id-2']}})
        self.mockclient.delete_firewall_policy.assert_called_once_with('5678')
        self.mockclient.show_firewall_policy.assert_not_called()

    def test_delete_failed(self):
        rsrc = self.create_firewall_policy()
        self.mockclient.delete_firewall_policy.side_effect = exceptions.NeutronClientException(status_code=400)
        scheduler.TaskRunner(rsrc.create)()
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.delete))
        self.assertEqual('NeutronClientException: resources.firewall_policy: An unknown exception occurred.', str(error))
        self.assertEqual((rsrc.DELETE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_firewall_policy.assert_called_once_with({'firewall_policy': {'name': 'test-firewall-policy', 'shared': True, 'audited': True, 'firewall_rules': ['rule-id-1', 'rule-id-2']}})
        self.mockclient.delete_firewall_policy.assert_called_once_with('5678')
        self.mockclient.show_firewall_policy.assert_not_called()

    def test_attribute(self):
        rsrc = self.create_firewall_policy()
        self.mockclient.show_firewall_policy.return_value = {'firewall_policy': {'audited': True, 'shared': True}}
        scheduler.TaskRunner(rsrc.create)()
        self.assertIs(True, rsrc.FnGetAtt('audited'))
        self.assertIs(True, rsrc.FnGetAtt('shared'))
        self.mockclient.create_firewall_policy.assert_called_once_with({'firewall_policy': {'name': 'test-firewall-policy', 'shared': True, 'audited': True, 'firewall_rules': ['rule-id-1', 'rule-id-2']}})
        self.mockclient.show_firewall_policy.assert_called_with('5678')

    def test_attribute_failed(self):
        rsrc = self.create_firewall_policy()
        scheduler.TaskRunner(rsrc.create)()
        error = self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'subnet_id')
        self.assertEqual('The Referenced Attribute (firewall_policy subnet_id) is incorrect.', str(error))
        self.mockclient.create_firewall_policy.assert_called_once_with({'firewall_policy': {'name': 'test-firewall-policy', 'shared': True, 'audited': True, 'firewall_rules': ['rule-id-1', 'rule-id-2']}})
        self.mockclient.show_firewall_policy.assert_not_called()

    def test_update(self):
        rsrc = self.create_firewall_policy()
        self.mockclient.update_firewall_policy.return_value = None
        scheduler.TaskRunner(rsrc.create)()
        props = self.tmpl['resources']['firewall_policy']['properties'].copy()
        props['firewall_rules'] = ['3', '4']
        update_template = rsrc.t.freeze(properties=props)
        scheduler.TaskRunner(rsrc.update, update_template)()
        self.mockclient.create_firewall_policy.assert_called_once_with({'firewall_policy': {'name': 'test-firewall-policy', 'shared': True, 'audited': True, 'firewall_rules': ['rule-id-1', 'rule-id-2']}})
        self.mockclient.update_firewall_policy.assert_called_once_with('5678', {'firewall_policy': {'firewall_rules': ['3', '4']}})