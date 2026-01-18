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
def test_resource_object_resource_properties_data(self):
    cfg.CONF.set_override('encrypt_parameters_and_properties', True)
    data = {'p1': 'i see', 'p2': 'good times, good times'}
    rpd_obj = rpd_object.ResourcePropertiesData().create_or_update(self.stack.context, data)
    with db_api.context_manager.writer.using(self.stack.context):
        rpd_db_obj = self.stack.context.session.get(models.ResourcePropertiesData, rpd_obj.id)
    res_obj1 = resource_objects.Resource().create(self.stack.context, {'stack_id': self.stack.id, 'uuid': str(uuid.uuid4()), 'rsrc_prop_data': rpd_db_obj})
    res_obj2 = resource_objects.Resource().create(self.stack.context, {'stack_id': self.stack.id, 'uuid': str(uuid.uuid4()), 'rsrc_prop_data_id': rpd_db_obj.id})
    ctx2 = utils.dummy_context()
    res_obj1 = resource_objects.Resource().get_obj(ctx2, res_obj1.id)
    res_obj2 = resource_objects.Resource().get_obj(ctx2, res_obj2.id)
    self.assertEqual(rpd_db_obj.id, res_obj1.rsrc_prop_data_id)
    self.assertEqual(res_obj1.rsrc_prop_data_id, res_obj2.rsrc_prop_data_id)
    self.assertEqual(data, res_obj1.properties_data)
    self.assertEqual(data, res_obj2.properties_data)