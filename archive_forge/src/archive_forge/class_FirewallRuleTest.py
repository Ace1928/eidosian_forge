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
class FirewallRuleTest(common.HeatTestCase):

    def setUp(self):
        super(FirewallRuleTest, self).setUp()
        self.mockclient = mock.Mock(spec=neutronclient.Client)
        self.patchobject(neutronclient, 'Client', return_value=self.mockclient)
        self.patchobject(neutron.NeutronClientPlugin, 'has_extension', return_value=True)

    def create_firewall_rule(self):
        self.mockclient.create_firewall_rule.return_value = {'firewall_rule': {'id': '5678'}}
        snippet = template_format.parse(firewall_rule_template)
        self.stack = utils.parse_stack(snippet)
        self.tmpl = snippet
        resource_defns = self.stack.t.resource_definitions(self.stack)
        return firewall.FirewallRule('firewall_rule', resource_defns['firewall_rule'], self.stack)

    def test_create(self):
        rsrc = self.create_firewall_rule()
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_firewall_rule.assert_called_once_with({'firewall_rule': {'name': 'test-firewall-rule', 'shared': True, 'action': 'allow', 'protocol': 'tcp', 'enabled': True, 'ip_version': '4'}})

    def test_validate_failed_with_string_None_protocol(self):
        snippet = template_format.parse(firewall_rule_template)
        stack = utils.parse_stack(snippet)
        rsrc = stack['firewall_rule']
        props = dict(rsrc.properties)
        props['protocol'] = 'None'
        rsrc.t = rsrc.t.freeze(properties=props)
        rsrc.reparse()
        self.assertRaises(exception.StackValidationFailed, rsrc.validate)

    def test_create_with_protocol_any(self):
        self.mockclient.create_firewall_rule.return_value = {'firewall_rule': {'id': '5678'}}
        snippet = template_format.parse(firewall_rule_template)
        snippet['resources']['firewall_rule']['properties']['protocol'] = 'any'
        stack = utils.parse_stack(snippet)
        rsrc = stack['firewall_rule']
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_firewall_rule.assert_called_once_with({'firewall_rule': {'name': 'test-firewall-rule', 'shared': True, 'action': 'allow', 'protocol': None, 'enabled': True, 'ip_version': '4'}})

    def test_create_failed(self):
        self.mockclient.create_firewall_rule.side_effect = exceptions.NeutronClientException()
        snippet = template_format.parse(firewall_rule_template)
        stack = utils.parse_stack(snippet)
        resource_defns = stack.t.resource_definitions(stack)
        rsrc = firewall.FirewallRule('firewall_rule', resource_defns['firewall_rule'], stack)
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
        self.assertEqual('NeutronClientException: resources.firewall_rule: An unknown exception occurred.', str(error))
        self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_firewall_rule.assert_called_once_with({'firewall_rule': {'name': 'test-firewall-rule', 'shared': True, 'action': 'allow', 'protocol': 'tcp', 'enabled': True, 'ip_version': '4'}})

    def test_delete(self):
        rsrc = self.create_firewall_rule()
        self.mockclient.delete_firewall_rule.return_value = None
        self.mockclient.show_firewall_rule.side_effect = exceptions.NeutronClientException(status_code=404)
        scheduler.TaskRunner(rsrc.create)()
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_firewall_rule.assert_called_once_with({'firewall_rule': {'name': 'test-firewall-rule', 'shared': True, 'action': 'allow', 'protocol': 'tcp', 'enabled': True, 'ip_version': '4'}})
        self.mockclient.delete_firewall_rule.assert_called_once_with('5678')
        self.mockclient.show_firewall_rule.assert_called_once_with('5678')

    def test_delete_already_gone(self):
        rsrc = self.create_firewall_rule()
        self.mockclient.delete_firewall_rule.side_effect = exceptions.NeutronClientException(status_code=404)
        scheduler.TaskRunner(rsrc.create)()
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_firewall_rule.assert_called_once_with({'firewall_rule': {'name': 'test-firewall-rule', 'shared': True, 'action': 'allow', 'protocol': 'tcp', 'enabled': True, 'ip_version': '4'}})
        self.mockclient.delete_firewall_rule.assert_called_once_with('5678')
        self.mockclient.show_firewall_rule.assert_not_called()

    def test_delete_failed(self):
        rsrc = self.create_firewall_rule()
        self.mockclient.delete_firewall_rule.side_effect = exceptions.NeutronClientException(status_code=400)
        scheduler.TaskRunner(rsrc.create)()
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.delete))
        self.assertEqual('NeutronClientException: resources.firewall_rule: An unknown exception occurred.', str(error))
        self.assertEqual((rsrc.DELETE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_firewall_rule.assert_called_once_with({'firewall_rule': {'name': 'test-firewall-rule', 'shared': True, 'action': 'allow', 'protocol': 'tcp', 'enabled': True, 'ip_version': '4'}})
        self.mockclient.delete_firewall_rule.assert_called_once_with('5678')
        self.mockclient.show_firewall_rule.assert_not_called()

    def test_attribute(self):
        rsrc = self.create_firewall_rule()
        self.mockclient.show_firewall_rule.return_value = {'firewall_rule': {'protocol': 'tcp', 'shared': True}}
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual('tcp', rsrc.FnGetAtt('protocol'))
        self.assertIs(True, rsrc.FnGetAtt('shared'))
        self.mockclient.create_firewall_rule.assert_called_once_with({'firewall_rule': {'name': 'test-firewall-rule', 'shared': True, 'action': 'allow', 'protocol': 'tcp', 'enabled': True, 'ip_version': '4'}})
        self.mockclient.show_firewall_rule.assert_called_with('5678')

    def test_attribute_failed(self):
        rsrc = self.create_firewall_rule()
        scheduler.TaskRunner(rsrc.create)()
        error = self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'subnet_id')
        self.assertEqual('The Referenced Attribute (firewall_rule subnet_id) is incorrect.', str(error))
        self.mockclient.create_firewall_rule.assert_called_once_with({'firewall_rule': {'name': 'test-firewall-rule', 'shared': True, 'action': 'allow', 'protocol': 'tcp', 'enabled': True, 'ip_version': '4'}})
        self.mockclient.show_firewall_rule.assert_not_called()

    def test_update(self):
        rsrc = self.create_firewall_rule()
        self.mockclient.update_firewall_rule.return_value = None
        scheduler.TaskRunner(rsrc.create)()
        props = self.tmpl['resources']['firewall_rule']['properties'].copy()
        props['protocol'] = 'icmp'
        update_template = rsrc.t.freeze(properties=props)
        scheduler.TaskRunner(rsrc.update, update_template)()
        self.mockclient.create_firewall_rule.assert_called_once_with({'firewall_rule': {'name': 'test-firewall-rule', 'shared': True, 'action': 'allow', 'protocol': 'tcp', 'enabled': True, 'ip_version': '4'}})
        self.mockclient.update_firewall_rule.assert_called_once_with('5678', {'firewall_rule': {'protocol': 'icmp'}})

    def test_update_protocol_to_any(self):
        rsrc = self.create_firewall_rule()
        self.mockclient.update_firewall_rule.return_value = None
        scheduler.TaskRunner(rsrc.create)()
        props = self.tmpl['resources']['firewall_rule']['properties'].copy()
        props['protocol'] = 'any'
        update_template = rsrc.t.freeze(properties=props)
        scheduler.TaskRunner(rsrc.update, update_template)()
        self.mockclient.create_firewall_rule.assert_called_once_with({'firewall_rule': {'name': 'test-firewall-rule', 'shared': True, 'action': 'allow', 'protocol': 'tcp', 'enabled': True, 'ip_version': '4'}})
        self.mockclient.update_firewall_rule.assert_called_once_with('5678', {'firewall_rule': {'protocol': None}})