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
def _test_instance_status_not_build_active(self, uncommon_status):
    return_server = self.fc.servers.list()[0]
    instance = self._setup_test_instance(return_server, 'in_sts_bld')
    instance.resource_id = '1234'
    status_calls = []

    def get_with_status(*args):
        if not status_calls:
            return_server.status = uncommon_status
        else:
            return_server.status = 'ACTIVE'
        status_calls.append(None)
        return return_server
    self.fc.servers.get = mock.Mock(side_effect=get_with_status)
    scheduler.TaskRunner(instance.create)()
    self.assertEqual((instance.CREATE, instance.COMPLETE), instance.state)
    self.assertGreaterEqual(self.fc.servers.get.call_count, 2)