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
class EnvironmentDuplicateTest(common.HeatTestCase):
    scenarios = [('same', dict(resource_type='test.yaml', expected_equal=True)), ('diff_temp', dict(resource_type='not.yaml', expected_equal=False)), ('diff_map', dict(resource_type='OS::Nova::Server', expected_equal=False)), ('diff_path', dict(resource_type='a/test.yaml', expected_equal=False))]

    def setUp(self):
        super(EnvironmentDuplicateTest, self).setUp(quieten_logging=False)

    def test_env_load(self):
        env_initial = {u'resource_registry': {u'OS::Test::Dummy': 'test.yaml'}}
        env = environment.Environment()
        env.load(env_initial)
        info = env.get_resource_info('OS::Test::Dummy', 'something')
        replace_log = 'Changing %s from %s to %s' % ('OS::Test::Dummy', 'test.yaml', self.resource_type)
        self.assertNotIn(replace_log, self.LOG.output)
        env_test = {u'resource_registry': {u'OS::Test::Dummy': self.resource_type}}
        env.load(env_test)
        if self.expected_equal:
            self.assertIs(info, env.get_resource_info('OS::Test::Dummy', 'my_fip'))
            self.assertNotIn(replace_log, self.LOG.output)
        else:
            self.assertIn(replace_log, self.LOG.output)
            self.assertNotEqual(info, env.get_resource_info('OS::Test::Dummy', 'my_fip'))

    def test_env_register_while_get_resource_info(self):
        env_test = {u'resource_registry': {u'OS::Test::Dummy': self.resource_type}}
        env = environment.Environment()
        env.load(env_test)
        env.get_resource_info('OS::Test::Dummy')
        self.assertEqual({'OS::Test::Dummy': self.resource_type, 'resources': {}}, env.user_env_as_dict().get(environment_format.RESOURCE_REGISTRY))
        env_test = {u'resource_registry': {u'resources': {u'test': {u'OS::Test::Dummy': self.resource_type}}}}
        env.load(env_test)
        env.get_resource_info('OS::Test::Dummy')
        self.assertEqual({u'OS::Test::Dummy': self.resource_type, 'resources': {u'test': {u'OS::Test::Dummy': self.resource_type}}}, env.user_env_as_dict().get(environment_format.RESOURCE_REGISTRY))