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
def test_register_with_path(self):
    yaml_env = '\n        resource_registry:\n          test::one: a.yaml\n          resources:\n            res_x:\n              test::two: b.yaml\n'
    env = environment.Environment(environment_format.parse(yaml_env))
    self.assertEqual('a.yaml', env.get_resource_info('test::one').value)
    self.assertEqual('b.yaml', env.get_resource_info('test::two', 'res_x').value)
    env2 = environment.Environment()
    env2.register_class('test::one', 'a.yaml', path=['test::one'])
    env2.register_class('test::two', 'b.yaml', path=['resources', 'res_x', 'test::two'])
    self.assertEqual(env.env_as_dict(), env2.env_as_dict())