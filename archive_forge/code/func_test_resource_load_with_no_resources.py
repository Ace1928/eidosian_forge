import collections
import datetime
import itertools
import json
import os
import sys
from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import short_id
from heat.common import timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import attributes
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import clients
from heat.engine import constraints
from heat.engine import dependencies
from heat.engine import environment
from heat.engine import node_data
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources.openstack.heat import none_resource
from heat.engine.resources.openstack.heat import test_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.engine import translation
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_object
from heat.objects import resource_properties_data as rpd_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
import neutronclient.common.exceptions as neutron_exp
def test_resource_load_with_no_resources(self):
    self.stack = parser.Stack(utils.dummy_context(), 'test_old_stack', template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'test_res': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': 'abc'}}}}))
    self.stack.store()
    snippet = rsrc_defn.ResourceDefinition('aresource', 'GenericResourceType')
    res = resource.Resource('aresource', snippet, self.stack)
    res.current_template_id = self.stack.t.id
    res.state_set('CREATE', 'IN_PROGRESS')
    self.stack.add_resource(res)
    origin_resources = self.stack.resources
    self.stack._resources = None
    loaded_res, res_owning_stack, stack = resource.Resource.load(self.stack.context, res.id, self.stack.current_traversal, False, {})
    self.assertEqual(origin_resources, stack.resources)
    self.assertEqual(loaded_res.id, res.id)
    self.assertEqual(self.stack.t, stack.t)