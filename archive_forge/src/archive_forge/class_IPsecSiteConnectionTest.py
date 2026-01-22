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
class IPsecSiteConnectionTest(common.HeatTestCase):
    IPSEC_SITE_CONNECTION_CONF = {'ipsec_site_connection': {'name': 'IPsecSiteConnection', 'description': 'My new VPN connection', 'peer_address': '172.24.4.233', 'peer_id': '172.24.4.233', 'peer_cidrs': ['10.2.0.0/24'], 'mtu': 1500, 'dpd': {'actions': 'hold', 'interval': 30, 'timeout': 120}, 'psk': 'secret', 'initiator': 'bi-directional', 'admin_state_up': True, 'ikepolicy_id': 'ike123', 'ipsecpolicy_id': 'ips123', 'vpnservice_id': 'vpn123'}}

    def setUp(self):
        super(IPsecSiteConnectionTest, self).setUp()
        self.mockclient = mock.Mock(spec=neutronclient.Client)
        self.patchobject(neutronclient, 'Client', return_value=self.mockclient)
        self.patchobject(neutron.NeutronClientPlugin, 'has_extension', return_value=True)

    def create_ipsec_site_connection(self):
        self.mockclient.create_ipsec_site_connection.return_value = {'ipsec_site_connection': {'id': 'con123'}}
        snippet = template_format.parse(ipsec_site_connection_template)
        self.stack = utils.parse_stack(snippet)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        return vpnservice.IPsecSiteConnection('ipsec_site_connection', resource_defns['IPsecSiteConnection'], self.stack)

    def test_create(self):
        rsrc = self.create_ipsec_site_connection()
        self.mockclient.show_ipsec_site_connection.return_value = {'ipsec_site_connection': {'status': 'ACTIVE'}}
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_ipsec_site_connection.assert_called_once_with(self.IPSEC_SITE_CONNECTION_CONF)
        self.mockclient.show_ipsec_site_connection.assert_called_once_with('con123')

    def test_create_failed(self):
        self.mockclient.create_ipsec_site_connection.side_effect = exceptions.NeutronClientException
        snippet = template_format.parse(ipsec_site_connection_template)
        self.stack = utils.parse_stack(snippet)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        rsrc = vpnservice.IPsecSiteConnection('ipsec_site_connection', resource_defns['IPsecSiteConnection'], self.stack)
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
        self.assertEqual('NeutronClientException: resources.ipsec_site_connection: An unknown exception occurred.', str(error))
        self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_ipsec_site_connection.assert_called_once_with(self.IPSEC_SITE_CONNECTION_CONF)

    def test_create_failed_error_status(self):
        cfg.CONF.set_override('action_retry_limit', 0)
        rsrc = self.create_ipsec_site_connection()
        self.mockclient.show_ipsec_site_connection.side_effect = [{'ipsec_site_connection': {'status': 'PENDING_CREATE'}}, {'ipsec_site_connection': {'status': 'ERROR'}}]
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
        self.assertEqual('ResourceInError: resources.ipsec_site_connection: Went to status ERROR due to "Error in IPsecSiteConnection"', str(error))
        self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_ipsec_site_connection.assert_called_once_with(self.IPSEC_SITE_CONNECTION_CONF)
        self.mockclient.show_ipsec_site_connection.assert_called_with('con123')

    def test_delete(self):
        rsrc = self.create_ipsec_site_connection()
        self.mockclient.show_ipsec_site_connection.side_effect = [{'ipsec_site_connection': {'status': 'ACTIVE'}}, exceptions.NeutronClientException(status_code=404)]
        self.mockclient.delete_ipsec_site_connection.return_value = None
        scheduler.TaskRunner(rsrc.create)()
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_ipsec_site_connection.assert_called_once_with(self.IPSEC_SITE_CONNECTION_CONF)
        self.mockclient.delete_ipsec_site_connection.assert_called_once_with('con123')
        self.mockclient.show_ipsec_site_connection.assert_called_with('con123')
        self.assertEqual(2, self.mockclient.show_ipsec_site_connection.call_count)

    def test_delete_already_gone(self):
        self.mockclient.show_ipsec_site_connection.return_value = {'ipsec_site_connection': {'status': 'ACTIVE'}}
        self.mockclient.delete_ipsec_site_connection.side_effect = exceptions.NeutronClientException(status_code=404)
        rsrc = self.create_ipsec_site_connection()
        scheduler.TaskRunner(rsrc.create)()
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mockclient.create_ipsec_site_connection.assert_called_once_with(self.IPSEC_SITE_CONNECTION_CONF)
        self.mockclient.show_ipsec_site_connection.assert_called_once_with('con123')
        self.mockclient.delete_ipsec_site_connection.assert_called_once_with('con123')

    def test_delete_failed(self):
        self.mockclient.show_ipsec_site_connection.return_value = {'ipsec_site_connection': {'status': 'ACTIVE'}}
        self.mockclient.delete_ipsec_site_connection.side_effect = exceptions.NeutronClientException(status_code=400)
        rsrc = self.create_ipsec_site_connection()
        scheduler.TaskRunner(rsrc.create)()
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.delete))
        self.assertEqual('NeutronClientException: resources.ipsec_site_connection: An unknown exception occurred.', str(error))
        self.assertEqual((rsrc.DELETE, rsrc.FAILED), rsrc.state)
        self.mockclient.create_ipsec_site_connection.assert_called_once_with(self.IPSEC_SITE_CONNECTION_CONF)
        self.mockclient.show_ipsec_site_connection.assert_called_once_with('con123')
        self.mockclient.delete_ipsec_site_connection.assert_called_once_with('con123')

    def test_attribute(self):
        rsrc = self.create_ipsec_site_connection()
        self.mockclient.show_ipsec_site_connection.return_value = {'ipsec_site_connection': {'status': 'ACTIVE'}}
        scheduler.TaskRunner(rsrc.create)()
        self.mockclient.show_ipsec_site_connection.return_value = self.IPSEC_SITE_CONNECTION_CONF
        self.assertEqual('IPsecSiteConnection', rsrc.FnGetAtt('name'))
        self.assertEqual('My new VPN connection', rsrc.FnGetAtt('description'))
        self.assertEqual('172.24.4.233', rsrc.FnGetAtt('peer_address'))
        self.assertEqual('172.24.4.233', rsrc.FnGetAtt('peer_id'))
        self.assertEqual(['10.2.0.0/24'], rsrc.FnGetAtt('peer_cidrs'))
        self.assertEqual('hold', rsrc.FnGetAtt('dpd')['actions'])
        self.assertEqual(30, rsrc.FnGetAtt('dpd')['interval'])
        self.assertEqual(120, rsrc.FnGetAtt('dpd')['timeout'])
        self.assertEqual('secret', rsrc.FnGetAtt('psk'))
        self.assertEqual('bi-directional', rsrc.FnGetAtt('initiator'))
        self.assertIs(True, rsrc.FnGetAtt('admin_state_up'))
        self.assertEqual('ike123', rsrc.FnGetAtt('ikepolicy_id'))
        self.assertEqual('ips123', rsrc.FnGetAtt('ipsecpolicy_id'))
        self.assertEqual('vpn123', rsrc.FnGetAtt('vpnservice_id'))
        self.mockclient.create_ipsec_site_connection.assert_called_once_with(self.IPSEC_SITE_CONNECTION_CONF)
        self.mockclient.show_ipsec_site_connection.assert_called_with('con123')

    def test_attribute_failed(self):
        rsrc = self.create_ipsec_site_connection()
        self.mockclient.show_ipsec_site_connection.return_value = {'ipsec_site_connection': {'status': 'ACTIVE'}}
        scheduler.TaskRunner(rsrc.create)()
        error = self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'non-existent_property')
        self.assertEqual('The Referenced Attribute (ipsec_site_connection non-existent_property) is incorrect.', str(error))
        self.mockclient.create_ipsec_site_connection.assert_called_once_with(self.IPSEC_SITE_CONNECTION_CONF)
        self.mockclient.show_ipsec_site_connection.assert_called_with('con123')

    def test_update(self):
        rsrc = self.create_ipsec_site_connection()
        self.mockclient.show_ipsec_site_connection.return_value = {'ipsec_site_connection': {'status': 'ACTIVE'}}
        self.mockclient.update_ipsec_site_connection.return_value = None
        scheduler.TaskRunner(rsrc.create)()
        props = dict(rsrc.properties)
        props['admin_state_up'] = False
        update_template = rsrc.t.freeze(properties=props)
        scheduler.TaskRunner(rsrc.update, update_template)()
        self.mockclient.create_ipsec_site_connection.assert_called_once_with(self.IPSEC_SITE_CONNECTION_CONF)
        self.mockclient.show_ipsec_site_connection.assert_called_with('con123')
        update = self.mockclient.update_ipsec_site_connection
        update.assert_called_once_with('con123', {'ipsec_site_connection': {'admin_state_up': False}})