import collections
import copy
import datetime
import json
import logging
import time
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from heat.common import context
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.db import api as db_api
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine import node_data
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import service
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.engine import update
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import stack as stack_object
from heat.objects import stack_tag as stack_tag_object
from heat.objects import user_creds as ucreds_object
from heat.tests import common
from heat.tests import fakes
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_parameters_created_encrypted_updated_decrypted(self):
    """Test stack loading with disabled parameter value validation."""
    tmpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            param1:\n                type: string\n                description: value1.\n            param2:\n                type: string\n                description: value2.\n                hidden: true\n        resources:\n            a_resource:\n                type: GenericResourceType\n        ')
    cfg.CONF.set_override('encrypt_parameters_and_properties', True)
    env1 = environment.Environment({'param1': 'foo', 'param2': 'bar'})
    self.stack = stack.Stack(self.ctx, 'test', template.Template(tmpl, env=env1))
    self.stack.store()
    cfg.CONF.set_override('encrypt_parameters_and_properties', False)
    loaded_stack = stack.Stack.load(self.ctx, stack_id=self.stack.id)
    loaded_stack.state_set(self.stack.CREATE, self.stack.COMPLETE, 'for_update')
    env2 = environment.Environment({'param1': 'foo', 'param2': 'new_bar'})
    new_stack = stack.Stack(self.ctx, 'test_update', template.Template(tmpl, env=env2))
    self.assertEqual(['param2'], loaded_stack.env.encrypted_param_names)
    loaded_stack.update(new_stack)
    self.assertEqual([], loaded_stack.env.encrypted_param_names)