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
class HookMatchTest(common.HeatTestCase):
    scenarios = [(hook_type, {'hook': hook_type}) for hook_type in environment.HOOK_TYPES]

    def test_plain_matches(self):
        other_hook = next((hook for hook in environment.HOOK_TYPES if hook != self.hook))
        resources = {u'a': {u'OS::Fruit': u'apples.yaml', u'hooks': [self.hook, other_hook]}, u'b': {u'OS::Food': u'fruity.yaml'}, u'nested': {u'res': {u'hooks': self.hook}}}
        registry = environment.ResourceRegistry(None, {})
        registry.load({u'OS::Fruit': u'apples.yaml', 'resources': resources})
        self.assertTrue(registry.matches_hook('a', self.hook))
        self.assertFalse(registry.matches_hook('b', self.hook))
        self.assertFalse(registry.matches_hook('OS::Fruit', self.hook))
        self.assertFalse(registry.matches_hook('res', self.hook))
        self.assertFalse(registry.matches_hook('unknown', self.hook))

    def test_wildcard_matches(self):
        other_hook = next((hook for hook in environment.HOOK_TYPES if hook != self.hook))
        resources = {u'prefix_*': {u'hooks': self.hook}, u'*_suffix': {u'hooks': self.hook}, u'*': {u'hooks': other_hook}}
        registry = environment.ResourceRegistry(None, {})
        registry.load({'resources': resources})
        self.assertTrue(registry.matches_hook('prefix_', self.hook))
        self.assertTrue(registry.matches_hook('prefix_some', self.hook))
        self.assertFalse(registry.matches_hook('some_prefix', self.hook))
        self.assertTrue(registry.matches_hook('_suffix', self.hook))
        self.assertTrue(registry.matches_hook('some_suffix', self.hook))
        self.assertFalse(registry.matches_hook('_suffix_blah', self.hook))
        self.assertTrue(registry.matches_hook('some_prefix', other_hook))
        self.assertTrue(registry.matches_hook('_suffix_blah', other_hook))

    def test_hook_types(self):
        resources = {u'hook': {u'hooks': self.hook}, u'not-hook': {u'hooks': [hook for hook in environment.HOOK_TYPES if hook != self.hook]}, u'all': {u'hooks': environment.HOOK_TYPES}}
        registry = environment.ResourceRegistry(None, {})
        registry.load({'resources': resources})
        self.assertTrue(registry.matches_hook('hook', self.hook))
        self.assertFalse(registry.matches_hook('not-hook', self.hook))
        self.assertTrue(registry.matches_hook('all', self.hook))