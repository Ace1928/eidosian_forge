from unittest import mock
from neutronclient.common import exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class NeutronL2GatewayTest(common.HeatTestCase):
    test_template = '\n    heat_template_version: queens\n    description: Template to test L2Gateway Neutron resource\n    resources:\n      l2gw:\n        type: OS::Neutron::L2Gateway\n        properties:\n          name: L2GW01\n          devices:\n            - device_name: switch01\n              interfaces:\n                - name: eth0\n                - name: eth1\n    '
    test_template_update = '\n    heat_template_version: queens\n    description: Template to test L2Gateway Neutron resource\n    resources:\n      l2gw:\n        type: OS::Neutron::L2Gateway\n        properties:\n          name: L2GW01\n          devices:\n            - device_name: switch01\n              interfaces:\n                - name: eth0\n                - name: eth1\n            - device_name: switch02\n              interfaces:\n                - name: eth5\n                - name: eth6\n    '
    test_template_with_seg = '\n    heat_template_version: queens\n    description: Template to test L2Gateway Neutron resource\n    resources:\n      l2gw:\n        type: OS::Neutron::L2Gateway\n        properties:\n          name: L2GW01\n          devices:\n            - device_name: switch01\n              interfaces:\n                - name: eth0\n                  segmentation_id:\n                    - 101\n                    - 102\n                    - 103\n                - name: eth1\n                  segmentation_id:\n                    - 101\n                    - 102\n                    - 103\n    '
    test_template_invalid_seg = '\n    heat_template_version: queens\n    description: Template to test L2Gateway Neutron resource\n    resources:\n      l2gw:\n        type: OS::Neutron::L2Gateway\n        properties:\n          name: L2GW01\n          devices:\n            - device_name: switch01\n              interfaces:\n                - name: eth0\n                  segmentation_id:\n                    - 101\n                    - 102\n                    - 103\n                - name: eth1\n    '
    mock_create_req = {'l2_gateway': {'name': 'L2GW01', 'devices': [{'device_name': 'switch01', 'interfaces': [{'name': 'eth0'}, {'name': 'eth1'}]}]}}
    mock_create_reply = {'l2_gateway': {'name': 'L2GW01', 'id': 'd3590f37-b072-4358-9719-71964d84a31c', 'tenant_id': '7ea656c7c9b8447494f33b0bc741d9e6', 'devices': [{'device_name': 'switch01', 'interfaces': [{'name': 'eth0'}, {'name': 'eth1'}]}]}}
    mock_update_req = {'l2_gateway': {'devices': [{'device_name': 'switch01', 'interfaces': [{'name': 'eth0'}, {'name': 'eth1'}]}, {'device_name': 'switch02', 'interfaces': [{'name': 'eth5'}, {'name': 'eth6'}]}]}}
    mock_update_reply = {'l2_gateway': {'name': 'L2GW01', 'id': 'd3590f37-b072-4358-9719-71964d84a31c', 'tenant_id': '7ea656c7c9b8447494f33b0bc741d9e6', 'devices': [{'device_name': 'switch01', 'interfaces': [{'name': 'eth0'}, {'name': 'eth1'}]}, {'device_name': 'switch02', 'interfaces': [{'name': 'eth5'}, {'name': 'eth6'}]}]}}
    mock_create_with_seg_req = {'l2_gateway': {'name': 'L2GW01', 'devices': [{'device_name': 'switch01', 'interfaces': [{'name': 'eth0', 'segmentation_id': [101, 102, 103]}, {'name': 'eth1', 'segmentation_id': [101, 102, 103]}]}]}}
    mock_create_with_seg_reply = {'l2_gateway': {'name': 'L2GW01', 'id': 'd3590f37-b072-4358-9719-71964d84a31c', 'tenant_id': '7ea656c7c9b8447494f33b0bc741d9e6', 'devices': [{'device_name': 'switch01', 'interfaces': [{'name': 'eth0', 'segmentation_id': ['101', '102', '103']}, {'name': 'eth1', 'segmentation_id': ['101', '102', '103']}]}]}}
    mock_create_invalid_seg_req = {'l2_gateway': {'name': 'L2GW01', 'devices': [{'device_name': 'switch01', 'interfaces': [{'name': 'eth0', 'segmentation_id': [101, 102, 103]}, {'name': 'eth1'}]}]}}
    mock_create_invalid_seg_reply = {'l2_gateway': {'name': 'L2GW01', 'id': 'd3590f37-b072-4358-9719-71964d84a31c', 'tenant_id': '7ea656c7c9b8447494f33b0bc741d9e6', 'devices': [{'device_name': 'switch01', 'interfaces': [{'name': 'eth0', 'segmentation_id': ['101', '102', '103']}, {'name': 'eth1'}]}]}}

    def setUp(self):
        super(NeutronL2GatewayTest, self).setUp()
        self.mockclient = mock.MagicMock()
        self.patchobject(neutronclient, 'Client', return_value=self.mockclient)
        self.patchobject(neutron.NeutronClientPlugin, 'has_extension', return_value=True)

    def _create_l2_gateway(self, hot, reply):
        self.mockclient.create_l2_gateway.return_value = reply
        self.mockclient.show_l2_gateway.return_value = reply
        template = template_format.parse(hot)
        self.stack = utils.parse_stack(template)
        scheduler.TaskRunner(self.stack.create)()
        self.l2gw_resource = self.stack['l2gw']

    def test_l2_gateway_create(self):
        self._create_l2_gateway(self.test_template, self.mock_create_reply)
        self.assertIsNone(self.l2gw_resource.validate())
        self.assertEqual((self.l2gw_resource.CREATE, self.l2gw_resource.COMPLETE), self.l2gw_resource.state)
        self.assertEqual('d3590f37-b072-4358-9719-71964d84a31c', self.l2gw_resource.FnGetRefId())
        self.mockclient.create_l2_gateway.assert_called_once_with(self.mock_create_req)

    def test_l2_gateway_update(self):
        self._create_l2_gateway(self.test_template, self.mock_create_reply)
        self.mockclient.update_l2_gateway.return_value = self.mock_update_reply
        self.mockclient.show_l2_gateway.return_value = self.mock_update_reply
        updated_tmpl = template_format.parse(self.test_template_update)
        updated_stack = utils.parse_stack(updated_tmpl)
        self.stack.update(updated_stack)
        ud_l2gw_resource = self.stack['l2gw']
        self.assertIsNone(ud_l2gw_resource.validate())
        self.assertEqual((ud_l2gw_resource.UPDATE, ud_l2gw_resource.COMPLETE), ud_l2gw_resource.state)
        self.assertEqual('d3590f37-b072-4358-9719-71964d84a31c', ud_l2gw_resource.FnGetRefId())
        self.mockclient.update_l2_gateway.assert_called_once_with('d3590f37-b072-4358-9719-71964d84a31c', self.mock_update_req)

    def test_l2_gateway_create_with_seg(self):
        self._create_l2_gateway(self.test_template_with_seg, self.mock_create_with_seg_reply)
        self.assertIsNone(self.l2gw_resource.validate())
        self.assertEqual((self.l2gw_resource.CREATE, self.l2gw_resource.COMPLETE), self.l2gw_resource.state)
        self.assertEqual('d3590f37-b072-4358-9719-71964d84a31c', self.l2gw_resource.FnGetRefId())
        self.mockclient.create_l2_gateway.assert_called_once_with(self.mock_create_with_seg_req)

    def test_l2_gateway_create_invalid_seg(self):
        self.mockclient.create_l2_gateway.side_effect = L2GatewaySegmentationRequired()
        template = template_format.parse(self.test_template_invalid_seg)
        self.stack = utils.parse_stack(template)
        scheduler.TaskRunner(self.stack.create)()
        self.l2gw_resource = self.stack['l2gw']
        self.assertIsNone(self.l2gw_resource.validate())
        self.assertEqual('Resource CREATE failed: L2GatewaySegmentationRequired: resources.l2gw: L2 gateway segmentation id must be consistent for all the interfaces', self.stack.status_reason)
        self.assertEqual((self.l2gw_resource.CREATE, self.l2gw_resource.FAILED), self.l2gw_resource.state)
        self.mockclient.create_l2_gateway.assert_called_once_with(self.mock_create_invalid_seg_req)

    def test_l2_gateway_delete(self):
        self._create_l2_gateway(self.test_template, self.mock_create_reply)
        self.stack.delete()
        self.mockclient.delete_l2_gateway.assert_called_with('d3590f37-b072-4358-9719-71964d84a31c')