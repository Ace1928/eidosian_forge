import copy
from unittest import mock
from oslo_config import cfg
from zunclient import exceptions as zc_exc
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import zun
from heat.engine.resources.openstack.zun import container
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _test_container_update_networks(self, new_networks):
    c = self._create_resource('container', self.rsrc_defn, self.stack)
    scheduler.TaskRunner(c.create)()
    t = template_format.parse(zun_template)
    new_t = copy.deepcopy(t)
    new_t['resources'][self.fake_name]['properties']['networks'] = new_networks
    rsrc_defns = template.Template(new_t).resource_definitions(self.stack)
    new_c = rsrc_defns[self.fake_name]
    sec_uuids = ['86c0f8ae-23a8-464f-8603-c54113ef5467']
    self.patchobject(neutron.NeutronClientPlugin, 'get_secgroup_uuids', return_value=sec_uuids)
    ifaces = [create_fake_iface(port='95e25541-d26a-478d-8f36-ae1c8f6b74dc', net='mynet', ip='10.0.0.4'), create_fake_iface(port='450abbc9-9b6d-4d6f-8c3a-c47ac34100ef', net='mynet2', ip='fe80::3'), create_fake_iface(port='myport', net='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', ip='21.22.23.24')]
    self.client.containers.network_list.return_value = ifaces
    scheduler.TaskRunner(c.update, new_c)()
    self.assertEqual((c.UPDATE, c.COMPLETE), c.state)
    self.client.containers.network_list.assert_called_once_with(self.resource_id)