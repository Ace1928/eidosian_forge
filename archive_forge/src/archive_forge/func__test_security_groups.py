import copy
from unittest import mock
import uuid
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients import progress
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _test_security_groups(self, instance, security_groups, sg='one', all_uuids=False, get_secgroup_raises=None):
    fake_groups_list, props = self._get_fake_properties(sg)
    if not all_uuids:
        self.nclient.list_security_groups = mock.Mock(return_value=fake_groups_list)
    self.patchobject(neutron.NeutronClientPlugin, 'network_id_from_subnet_id', return_value='fake_network_id')
    if get_secgroup_raises:
        self.assertRaises(get_secgroup_raises, instance._build_nics, None, security_groups=security_groups, subnet_id='fake_subnet_id')
    else:
        self.nclient.create_port = mock.Mock(return_value={'port': {'id': 'fake_port_id'}})
        self.stub_keystoneclient()
        self.assertEqual([{'port-id': 'fake_port_id'}], instance._build_nics(None, security_groups=security_groups, subnet_id='fake_subnet_id'))
        self.nclient.create_port.assert_called_with({'port': props})
    if not all_uuids:
        self.nclient.list_security_groups.assert_called_once_with(project_id=mock.ANY)