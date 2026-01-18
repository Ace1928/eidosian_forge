import os.path
from unittest import mock
import fixtures
from oslo_config import cfg
from heat.common import environment_format
from heat.common import exception
from heat.engine import environment
from heat.engine import resources
from heat.engine.resources.aws.ec2 import instance
from heat.engine.resources.openstack.nova import server
from heat.engine import support
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
def test_resource_sort_order_len(self):
    new_env = {u'resource_registry': {u'resources': {u'my_fip': {u'OS::Networking::FloatingIP': 'ip.yaml'}}}, u'OS::Networking::FloatingIP': 'OS::Nova::FloatingIP'}
    env = environment.Environment(new_env)
    self.assertEqual('ip.yaml', env.get_resource_info('OS::Networking::FloatingIP', 'my_fip').value)