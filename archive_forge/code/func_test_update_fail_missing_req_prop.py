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
def test_update_fail_missing_req_prop(self):
    tmpl = rsrc_defn.ResourceDefinition('test_resource', 'GenericResourceType', {'Foo': 'abc'})
    res = generic_rsrc.ResourceWithRequiredProps('test_resource', tmpl, self.stack)
    res.update_allowed_properties = ('Foo',)
    scheduler.TaskRunner(res.create)()
    self.assertEqual((res.CREATE, res.COMPLETE), res.state)
    utmpl = rsrc_defn.ResourceDefinition('test_resource', 'GenericResourceType', {})
    updater = scheduler.TaskRunner(res.update, utmpl)
    self.assertRaises(exception.ResourceFailure, updater)
    self.assertEqual((res.UPDATE, res.FAILED), res.state)