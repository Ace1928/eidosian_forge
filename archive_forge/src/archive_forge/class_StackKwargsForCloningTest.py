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
class StackKwargsForCloningTest(common.HeatTestCase):
    scenarios = [('default', dict(keep_status=False, only_db=False, keep_tags=False, not_included=['action', 'status', 'status_reason', 'tags'])), ('only_db', dict(keep_status=False, only_db=True, keep_tags=False, not_included=['action', 'status', 'status_reason', 'strict_validate', 'tags'])), ('keep_status', dict(keep_status=True, only_db=False, keep_tags=False, not_included=['tags'])), ('status_db', dict(keep_status=True, only_db=True, keep_tags=False, not_included=['strict_validate', 'tags'])), ('keep_tags', dict(keep_status=False, only_db=False, keep_tags=True, not_included=['action', 'status', 'status_reason']))]

    def test_kwargs(self):
        tmpl = template.Template(copy.deepcopy(empty_template))
        ctx = utils.dummy_context()
        test_data = dict(action='x', status='y', status_reason='z', timeout_mins=33, disable_rollback=True, parent_resource='fred', owner_id=32, stack_user_project_id=569, user_creds_id=123, tenant_id='some-uuid', username='jo', nested_depth=3, strict_validate=True, convergence=False, current_traversal=45, tags=['tag1', 'tag2'])
        db_map = {'parent_resource': 'parent_resource_name', 'tenant_id': 'tenant', 'timeout_mins': 'timeout'}
        test_db_data = {}
        for key in test_data:
            dbkey = db_map.get(key, key)
            test_db_data[dbkey] = test_data[key]
        self.stack = stack.Stack(ctx, utils.random_name(), tmpl, **test_data)
        res = self.stack.get_kwargs_for_cloning(keep_status=self.keep_status, only_db=self.only_db, keep_tags=self.keep_tags)
        for key in self.not_included:
            self.assertNotIn(key, res)
        for key in test_data:
            if key not in self.not_included:
                dbkey = db_map.get(key, key)
                if self.only_db:
                    self.assertEqual(test_data[key], res[dbkey])
                else:
                    self.assertEqual(test_data[key], res[key])
        if not self.only_db:
            stack.Stack(ctx, utils.random_name(), tmpl, **res)