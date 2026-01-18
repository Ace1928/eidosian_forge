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
def test_properties_data_stored_encrypted_decrypted_on_load(self):
    cfg.CONF.set_override('encrypt_parameters_and_properties', True)
    tmpl = rsrc_defn.ResourceDefinition('test_resource', 'Foo')
    stored_properties_data = {'prop1': 'string', 'prop2': {'a': 'dict'}, 'prop3': 1, 'prop4': ['a', 'list'], 'prop5': True}
    res = generic_rsrc.GenericResource('test_res_enc', tmpl, self.stack)
    res._stored_properties_data = stored_properties_data
    res._rsrc_prop_data = None
    res.store()
    db_res = db_api.resource_get(res.context, res.id)
    self.assertNotEqual('string', db_res.rsrc_prop_data.data['prop1'])
    res = generic_rsrc.GenericResource('test_res_enc', tmpl, self.stack)
    res._stored_properties_data = stored_properties_data
    res.state_set(res.CREATE, res.IN_PROGRESS, 'test_store')
    db_res = db_api.resource_get(res.context, res.id)
    self.assertNotEqual('string', db_res.rsrc_prop_data.data['prop1'])
    res_obj = resource_objects.Resource.get_obj(res.context, res.id)
    self.assertEqual('string', res_obj.properties_data['prop1'])
    res_objs = resource_objects.Resource.get_all_by_stack(res.context, self.stack.id)
    res_obj = res_objs['test_res_enc']
    self.assertEqual('string', res_obj.properties_data['prop1'])
    res_obj = resource_objects.Resource.get_obj(res.context, res.id)
    res_obj.refresh()
    self.assertEqual('string', res_obj.properties_data['prop1'])