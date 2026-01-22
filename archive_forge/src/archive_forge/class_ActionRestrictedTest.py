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
class ActionRestrictedTest(common.HeatTestCase):

    def test_plain_matches(self):
        resources = {u'a': {u'OS::Fruit': u'apples.yaml', u'restricted_actions': [u'update', u'replace']}, u'b': {u'OS::Food': u'fruity.yaml'}, u'nested': {u'res': {u'restricted_actions': 'update'}}}
        registry = environment.ResourceRegistry(None, {})
        registry.load({u'OS::Fruit': u'apples.yaml', 'resources': resources})
        self.assertIn(environment.UPDATE, registry.get_rsrc_restricted_actions('a'))
        self.assertNotIn(environment.UPDATE, registry.get_rsrc_restricted_actions('b'))
        self.assertNotIn(environment.UPDATE, registry.get_rsrc_restricted_actions('OS::Fruit'))
        self.assertNotIn(environment.UPDATE, registry.get_rsrc_restricted_actions('res'))
        self.assertNotIn(environment.UPDATE, registry.get_rsrc_restricted_actions('unknown'))

    def test_wildcard_matches(self):
        resources = {u'prefix_*': {u'restricted_actions': 'update'}, u'*_suffix': {u'restricted_actions': 'update'}, u'*': {u'restricted_actions': 'replace'}}
        registry = environment.ResourceRegistry(None, {})
        registry.load({'resources': resources})
        self.assertIn(environment.UPDATE, registry.get_rsrc_restricted_actions('prefix_'))
        self.assertIn(environment.UPDATE, registry.get_rsrc_restricted_actions('prefix_some'))
        self.assertNotIn(environment.UPDATE, registry.get_rsrc_restricted_actions('some_prefix'))
        self.assertIn(environment.UPDATE, registry.get_rsrc_restricted_actions('_suffix'))
        self.assertIn(environment.UPDATE, registry.get_rsrc_restricted_actions('some_suffix'))
        self.assertNotIn(environment.UPDATE, registry.get_rsrc_restricted_actions('_suffix_blah'))
        self.assertIn(environment.REPLACE, registry.get_rsrc_restricted_actions('some_prefix'))
        self.assertIn(environment.REPLACE, registry.get_rsrc_restricted_actions('_suffix_blah'))

    def test_restricted_action_types(self):
        resources = {u'update': {u'restricted_actions': 'update'}, u'replace': {u'restricted_actions': 'replace'}, u'all': {u'restricted_actions': ['update', 'replace']}}
        registry = environment.ResourceRegistry(None, {})
        registry.load({'resources': resources})
        self.assertIn(environment.UPDATE, registry.get_rsrc_restricted_actions('update'))
        self.assertNotIn(environment.UPDATE, registry.get_rsrc_restricted_actions('replace'))
        self.assertIn(environment.REPLACE, registry.get_rsrc_restricted_actions('replace'))
        self.assertNotIn(environment.REPLACE, registry.get_rsrc_restricted_actions('update'))
        self.assertIn(environment.UPDATE, registry.get_rsrc_restricted_actions('all'))
        self.assertIn(environment.REPLACE, registry.get_rsrc_restricted_actions('all'))