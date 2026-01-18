from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_port_security_enabled(self):
    t = template_format.parse(neutron_port_security_template)
    stack = utils.parse_stack(t)
    self.find_mock.return_value = 'abcd1234'
    self.create_mock.return_value = {'port': {'status': 'BUILD', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}
    self.port_show_mock.return_value = {'port': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}
    port = stack['port']
    scheduler.TaskRunner(port.create)()
    self.create_mock.assert_called_once_with({'port': {'network_id': u'abcd1234', 'port_security_enabled': False, 'name': utils.PhysName(stack.name, 'port'), 'admin_state_up': True, 'binding:vnic_type': 'normal', 'device_id': '', 'device_owner': ''}})