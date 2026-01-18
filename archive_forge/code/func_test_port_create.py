import copy
from unittest import mock
from ironicclient.common.apiclient import exceptions as ic_exc
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import ironic as ic
from heat.engine import resource
from heat.engine.resources.openstack.ironic import port
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_port_create(self):
    b = self._create_resource('port', self.rsrc_defn, self.stack)
    self.assertEqual(self.fake_node_name, b.properties.get(port.Port.NODE))
    self.assertEqual(self.fake_address, b.properties.get(port.Port.ADDRESS))
    self.assertEqual(self.fake_portgroup_name, b.properties.get(port.Port.PORTGROUP))
    self.assertEqual(self.fake_local_link_connection, b.properties.get(port.Port.LOCAL_LINK_CONNECTION))
    self.assertEqual(self.fake_pxe_enabled, b.properties.get(port.Port.PXE_ENABLED))
    self.assertEqual(self.fake_physical_network, b.properties.get(port.Port.PHYSICAL_NETWORK))
    self.assertEqual(self.fake_extra, b.properties.get(port.Port.EXTRA))
    self.assertEqual(self.fake_is_smartnic, b.properties.get(port.Port.IS_SMARTNIC))
    scheduler.TaskRunner(b.create)()
    self.assertEqual(self.resource_id, b.resource_id)
    expected = [mock.call(self.fake_node_name), mock.call(self.fake_node_uuid)]
    self.assertEqual(expected, self.m_fgn.call_args_list)
    expected = [mock.call(self.fake_portgroup_name), mock.call(self.fake_portgroup_uuid)]
    self.assertEqual(expected, self.m_fgpg.call_args_list)
    self.client.port.create.assert_called_once_with(address=self.fake_address, extra=self.fake_extra, is_smartnic=self.fake_is_smartnic, local_link_connection=self.fake_local_link_connection, node_uuid=self.fake_node_uuid, physical_network=self.fake_physical_network, portgroup_uuid=self.fake_portgroup_uuid, pxe_enabled=self.fake_pxe_enabled)