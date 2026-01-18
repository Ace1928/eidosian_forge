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
def test_verify_check_conditions(self):
    valid_foos = ['foo1', 'foo2']
    checks = [{'attr': 'foo1', 'expected': 'bar1', 'current': 'baz1'}, {'attr': 'foo2', 'expected': valid_foos, 'current': 'foo2'}, {'attr': 'foo3', 'expected': 'bar3', 'current': 'baz3'}, {'attr': 'foo4', 'expected': 'foo4', 'current': 'foo4'}, {'attr': 'foo5', 'expected': valid_foos, 'current': 'baz5'}]
    tmpl = rsrc_defn.ResourceDefinition('test_res', 'GenericResourceType')
    res = generic_rsrc.ResourceWithProps('test_res', tmpl, self.stack)
    exc = self.assertRaises(exception.Error, res._verify_check_conditions, checks)
    exc_text = str(exc)
    self.assertNotIn("'foo2':", exc_text)
    self.assertNotIn("'foo4':", exc_text)
    self.assertIn("'foo1': expected 'bar1', got 'baz1'", exc_text)
    self.assertIn("'foo3': expected 'bar3', got 'baz3'", exc_text)
    self.assertIn("'foo5': expected '['foo1', 'foo2']', got 'baz5'", exc_text)