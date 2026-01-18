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
@mock.patch.object(translation, 'resolve_and_find')
@mock.patch.object(parser.Stack, 'db_resource_get')
@mock.patch.object(resource.Resource, '_load_data')
@mock.patch.object(resource.Resource, 'translate_properties')
def test_stack_resources(self, mock_translate, mock_load, mock_db_get, mock_resolve):
    tpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'A': {'Type': 'ResourceWithPropsType', 'Properties': {'Foo': 'abc'}}}}
    stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(tpl))
    stack.store()
    mock_db_get.return_value = None
    self.assertEqual(1, len(stack.resources))
    self.assertEqual(1, mock_translate.call_count)
    self.assertEqual(0, mock_load.call_count)
    stack._resources = None
    mock_db_get.return_value = mock.Mock()
    self.assertEqual(1, len(stack.resources))
    self.assertEqual(2, mock_translate.call_count)
    self.assertEqual(1, mock_load.call_count)
    self.assertEqual(0, mock_resolve.call_count)