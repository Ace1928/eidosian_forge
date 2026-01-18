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