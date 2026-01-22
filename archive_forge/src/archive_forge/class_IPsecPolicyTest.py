import copy
from unittest import mock
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import vpnservice
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class IPsecPolicyTest(common.HeatTestCase):
    IPSEC_POLICY_CONF = {'ipsecpolicy': {'name': 'IPsecPolicy', 'description': 'My new IPsec policy', 'transform_protocol': 'esp', 'encapsulation_mode': 'tunnel', 'auth_algorithm': 'sha1', 'encryption_algorithm': '3des', 'lifetime': {'units': 'seconds', 'value': 3600}, 'pfs': 'group5'}}

    def setUp(self):
        super(IPsecPolicyTest, self).setUp()
        self.mockclient = mock.Mock(spec=neutronclient.Client)
        self.patchobject(neutronclient, 'Client', return_value=self.mockclient)
        self.patchobject(neutron.NeutronClientPlugin, 'has_extension', return_value=True)

    def create_ipsecpolicy(self):
        self.mockclient.create_ipsecpolicy.return_value = {'ipsecpolicy': {'id': 'ips123'}}
        snippet = template_format.parse(ipsecpolicy_template)
        self.stack = utils.parse_stack(snippet)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        return vpnservice.IPsecPolicy('ipsecpolicy', resource_defns['IPsecPolicy'], self.stack)

    def test_create(self):
        rsrc = self.create_ipsecpolicy()
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_ipsecpolicy.assert_called_once_with(self.IPSEC_POLICY_CONF)

    def test_create_failed(self):
        self.mockclient.create_ipsecpolicy.side_effect = exceptions.NeutronClientException
        snippet = template_format.parse(ipsecpolicy_template)
        self.stack = utils.parse_stack(snippet)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        rsrc = vpnservice.IPsecPolicy('ipsecpolicy', resource_defns['IPsecPolicy'], self.stack)
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
        self.assertEqual('NeutronClientException: resources.ipsecpolicy: An unknown exception occurred.', str(error))
        self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_ipsecpolicy.assert_called_once_with(self.IPSEC_POLICY_CONF)

    def test_delete(self):
        rsrc = self.create_ipsecpolicy()
        self.mockclient.delete_ipsecpolicy.return_value = None
        self.mockclient.show_ipsecpolicy.side_effect = exceptions.NeutronClientException(status_code=404)
        scheduler.TaskRunner(rsrc.create)()
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_ipsecpolicy.assert_called_once_with(self.IPSEC_POLICY_CONF)
        self.mockclient.delete_ipsecpolicy.assert_called_once_with('ips123')
        self.mockclient.show_ipsecpolicy.assert_called_once_with('ips123')

    def test_delete_already_gone(self):
        rsrc = self.create_ipsecpolicy()
        self.mockclient.delete_ipsecpolicy.side_effect = exceptions.NeutronClientException(status_code=404)
        scheduler.TaskRunner(rsrc.create)()
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_ipsecpolicy.assert_called_once_with(self.IPSEC_POLICY_CONF)
        self.mockclient.delete_ipsecpolicy.assert_called_once_with('ips123')
        self.mockclient.show_ipsecpolicy.assert_not_called()

    def test_delete_failed(self):
        rsrc = self.create_ipsecpolicy()
        self.mockclient.delete_ipsecpolicy.side_effect = exceptions.NeutronClientException(status_code=400)
        scheduler.TaskRunner(rsrc.create)()
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.delete))
        self.assertEqual('NeutronClientException: resources.ipsecpolicy: An unknown exception occurred.', str(error))
        self.assertEqual((rsrc.DELETE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_ipsecpolicy.assert_called_once_with(self.IPSEC_POLICY_CONF)
        self.mockclient.delete_ipsecpolicy.assert_called_once_with('ips123')
        self.mockclient.show_ipsecpolicy.assert_not_called()

    def test_attribute(self):
        rsrc = self.create_ipsecpolicy()
        self.mockclient.show_ipsecpolicy.return_value = self.IPSEC_POLICY_CONF
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual('IPsecPolicy', rsrc.FnGetAtt('name'))
        self.assertEqual('My new IPsec policy', rsrc.FnGetAtt('description'))
        self.assertEqual('esp', rsrc.FnGetAtt('transform_protocol'))
        self.assertEqual('tunnel', rsrc.FnGetAtt('encapsulation_mode'))
        self.assertEqual('sha1', rsrc.FnGetAtt('auth_algorithm'))
        self.assertEqual('3des', rsrc.FnGetAtt('encryption_algorithm'))
        self.assertEqual('seconds', rsrc.FnGetAtt('lifetime')['units'])
        self.assertEqual(3600, rsrc.FnGetAtt('lifetime')['value'])
        self.assertEqual('group5', rsrc.FnGetAtt('pfs'))
        self.mockclient.create_ipsecpolicy.assert_called_once_with(self.IPSEC_POLICY_CONF)
        self.mockclient.show_ipsecpolicy.assert_called_with('ips123')

    def test_attribute_failed(self):
        rsrc = self.create_ipsecpolicy()
        scheduler.TaskRunner(rsrc.create)()
        error = self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'non-existent_property')
        self.assertEqual('The Referenced Attribute (ipsecpolicy non-existent_property) is incorrect.', str(error))
        self.mockclient.create_ipsecpolicy.assert_called_once_with(self.IPSEC_POLICY_CONF)
        self.mockclient.show_ipsecpolicy.assert_not_called()

    def test_update(self):
        rsrc = self.create_ipsecpolicy()
        self.mockclient.update_ipsecpolicy.return_value = None
        scheduler.TaskRunner(rsrc.create)()
        update_template = copy.deepcopy(rsrc.t)
        props = dict(rsrc.properties)
        props['name'] = 'New IPsecPolicy'
        update_template = rsrc.t.freeze(properties=props)
        scheduler.TaskRunner(rsrc.update, update_template)()
        self.mockclient.create_ipsecpolicy.assert_called_once_with(self.IPSEC_POLICY_CONF)
        self.mockclient.update_ipsecpolicy.assert_called_once_with('ips123', {'ipsecpolicy': {'name': 'New IPsecPolicy'}})
        self.mockclient.show_ipsecpolicy.assert_not_called()