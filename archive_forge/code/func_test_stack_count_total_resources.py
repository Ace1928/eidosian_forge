import datetime
import json
import time
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_utils import timeutils
from sqlalchemy import orm
from sqlalchemy.orm import exc
from sqlalchemy.orm import session
from heat.common import context
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.db import api as db_api
from heat.db import models
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import resource as rsrc
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.engine import template_files
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_stack_count_total_resources(self):

    def add_resources(stack, count, root_stack_id):
        for i in range(count):
            create_resource(self.ctx, stack, False, name='%s-%s' % (stack.name, i), root_stack_id=root_stack_id)
    root = create_stack(self.ctx, self.template, self.user_creds, name='root stack')
    s_1 = create_stack(self.ctx, self.template, self.user_creds, name='s_1', owner_id=root.id)
    s_1_1 = create_stack(self.ctx, self.template, self.user_creds, name='s_1_1', owner_id=s_1.id)
    s_1_2 = create_stack(self.ctx, self.template, self.user_creds, name='s_1_2', owner_id=s_1.id)
    s_1_3 = create_stack(self.ctx, self.template, self.user_creds, name='s_1_3', owner_id=s_1.id)
    s_2 = create_stack(self.ctx, self.template, self.user_creds, name='s_2', owner_id=root.id)
    s_2_1 = create_stack(self.ctx, self.template, self.user_creds, name='s_2_1', owner_id=s_2.id)
    s_2_1_1 = create_stack(self.ctx, self.template, self.user_creds, name='s_2_1_1', owner_id=s_2_1.id)
    s_2_1_1_1 = create_stack(self.ctx, self.template, self.user_creds, name='s_2_1_1_1', owner_id=s_2_1_1.id)
    s_3 = create_stack(self.ctx, self.template, self.user_creds, name='s_3', owner_id=root.id)
    s_4 = create_stack(self.ctx, self.template, self.user_creds, name='s_4', owner_id=root.id)
    add_resources(root, 3, root.id)
    add_resources(s_1, 2, root.id)
    add_resources(s_1_1, 4, root.id)
    add_resources(s_1_2, 5, root.id)
    add_resources(s_1_3, 6, root.id)
    add_resources(s_2, 1, root.id)
    add_resources(s_2_1_1_1, 1, root.id)
    add_resources(s_3, 4, root.id)
    self.assertEqual(26, db_api.stack_count_total_resources(self.ctx, root.id))
    self.assertEqual(0, db_api.stack_count_total_resources(self.ctx, s_4.id))
    self.assertEqual(0, db_api.stack_count_total_resources(self.ctx, 'asdf'))
    self.assertEqual(0, db_api.stack_count_total_resources(self.ctx, None))