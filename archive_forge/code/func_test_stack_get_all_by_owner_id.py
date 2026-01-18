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
def test_stack_get_all_by_owner_id(self):
    parent_stack1 = create_stack(self.ctx, self.template, self.user_creds)
    parent_stack2 = create_stack(self.ctx, self.template, self.user_creds)
    values = [{'owner_id': parent_stack1.id}, {'owner_id': parent_stack1.id}, {'owner_id': parent_stack2.id}, {'owner_id': parent_stack2.id}]
    [create_stack(self.ctx, self.template, self.user_creds, **val) for val in values]
    stack1_children = db_api.stack_get_all_by_owner_id(self.ctx, parent_stack1.id)
    self.assertEqual(2, len(stack1_children))
    stack2_children = db_api.stack_get_all_by_owner_id(self.ctx, parent_stack2.id)
    self.assertEqual(2, len(stack2_children))