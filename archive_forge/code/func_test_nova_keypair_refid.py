import copy
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine.resources.openstack.nova import keypair
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_nova_keypair_refid(self):
    stack = utils.parse_stack(self.kp_template)
    rsrc = stack['kp']
    rsrc.resource_id = 'xyz'
    self.assertEqual('xyz', rsrc.FnGetRefId())