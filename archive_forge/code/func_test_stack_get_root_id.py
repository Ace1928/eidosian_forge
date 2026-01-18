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
def test_stack_get_root_id(self):
    root = create_stack(self.ctx, self.template, self.user_creds, name='root stack')
    child_1 = create_stack(self.ctx, self.template, self.user_creds, name='child 1 stack', owner_id=root.id)
    child_2 = create_stack(self.ctx, self.template, self.user_creds, name='child 2 stack', owner_id=child_1.id)
    child_3 = create_stack(self.ctx, self.template, self.user_creds, name='child 3 stack', owner_id=child_2.id)
    self.assertEqual(root.id, db_api.stack_get_root_id(self.ctx, child_3.id))
    self.assertEqual(root.id, db_api.stack_get_root_id(self.ctx, child_2.id))
    self.assertEqual(root.id, db_api.stack_get_root_id(self.ctx, root.id))
    self.assertEqual(root.id, db_api.stack_get_root_id(self.ctx, child_1.id))
    self.assertIsNone(db_api.stack_get_root_id(self.ctx, 'non existent stack'))