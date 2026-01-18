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
def test_params_flat(self):
    new_params = {'foo': 'bar', 'tester': 'Yes'}
    penv = environment.Environment()
    expected = {'parameters': new_params, 'encrypted_param_names': [], 'parameter_defaults': {}, 'event_sinks': [], 'resource_registry': {'resources': {}}}
    cenv = environment.get_child_environment(penv, new_params)
    self.assertEqual(expected, cenv.env_as_dict())