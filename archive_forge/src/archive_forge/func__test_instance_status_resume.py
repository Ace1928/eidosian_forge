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
def _test_instance_status_resume(self, name, state=('SUSPEND', 'COMPLETE')):
    return_server = self.fc.servers.list()[1]
    instance = self._create_test_instance(return_server, name)
    instance.resource_id = '1234'
    instance.state_set(state[0], state[1])
    d1 = {'server': self.fc.client.get_servers_detail()[1]['servers'][0]}
    d2 = copy.deepcopy(d1)
    d1['server']['status'] = 'SUSPENDED'
    d2['server']['status'] = 'ACTIVE'
    self.fc.client.get_servers_1234 = mock.Mock(side_effect=[(200, d1), (200, d1), (200, d2)])
    instance.state_set(instance.SUSPEND, instance.COMPLETE)
    scheduler.TaskRunner(instance.resume)()
    self.assertEqual((instance.RESUME, instance.COMPLETE), instance.state)
    self.assertEqual(3, self.fc.client.get_servers_1234.call_count)