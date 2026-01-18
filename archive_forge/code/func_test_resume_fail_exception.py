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
def test_resume_fail_exception(self):
    tmpl = rsrc_defn.ResourceDefinition('test_resource', 'GenericResourceType', {'Foo': 'abc'})
    res = generic_rsrc.ResourceWithProps('test_resource', tmpl, self.stack)
    scheduler.TaskRunner(res.create)()
    self.assertEqual((res.CREATE, res.COMPLETE), res.state)

    class FakeExc(Exception):
        pass
    self.patchobject(generic_rsrc.ResourceWithProps, 'handle_resume', side_effect=FakeExc)
    res.state_set(res.SUSPEND, res.COMPLETE)
    resume = scheduler.TaskRunner(res.resume)
    self.assertRaises(exception.ResourceFailure, resume)
    self.assertEqual((res.RESUME, res.FAILED), res.state)
    self.assertEqual(1, generic_rsrc.ResourceWithProps.handle_resume.call_count)