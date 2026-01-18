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
def test_post_delete_hook_call(self):
    self.stack.env.registry.load({'resources': {'res': {'hooks': 'post-delete'}}})
    snippet = rsrc_defn.ResourceDefinition('res', 'GenericResourceType')
    res = resource.Resource('res', snippet, self.stack)
    res.id = '1234'
    res.uuid = uuid.uuid4()
    res.action = 'CREATE'
    self.stack.action = 'DELETE'
    res.store = mock.Mock()
    task = scheduler.TaskRunner(res.delete)
    task.start()
    task.step()
    self.assertTrue(res.has_hook('post-delete'))
    res.signal(details={'unset_hook': 'post-delete'})
    task.run_to_completion()
    self.assertEqual((res.DELETE, res.COMPLETE), res.state)