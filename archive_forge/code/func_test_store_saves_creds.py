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
def test_store_saves_creds(self):
    """A user_creds entry is created on first stack store."""
    cfg.CONF.set_default('deferred_auth_method', 'password')
    self.stack = stack.Stack(self.ctx, 'creds_stack', self.tmpl)
    self.stack.store()
    db_stack = stack_object.Stack.get_by_id(self.ctx, self.stack.id)
    user_creds_id = db_stack.user_creds_id
    self.assertIsNotNone(user_creds_id)
    user_creds = ucreds_object.UserCreds.get_by_id(self.ctx, user_creds_id)
    self.assertEqual(self.ctx.username, user_creds.get('username'))
    self.assertEqual(self.ctx.password, user_creds.get('password'))
    self.assertIsNone(user_creds.get('trust_id'))
    self.assertIsNone(user_creds.get('trustor_user_id'))
    expected_context = context.RequestContext.from_dict(self.ctx.to_dict())
    expected_context.auth_token = None
    stored_context = self.stack.stored_context().to_dict()
    self.assertEqual(expected_context.to_dict(), stored_context)
    self.stack.store()
    self.assertEqual(user_creds_id, db_stack.user_creds_id)