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
def test_stack_get_all_show_nested(self):
    stack1 = self._setup_test_stack('neststack_get_all_1', UUID1)[1]
    stack2 = self._setup_test_stack('neststack_get_all_2', UUID2, owner_id=stack1.id)[1]
    stack3 = self._setup_test_stack('neststack_get_all_1*', UUID3, owner_id=stack1.id, backup=True)[1]
    st_db = db_api.stack_get_all(self.ctx)
    self.assertEqual(1, len(st_db))
    self.assertEqual(stack1.id, st_db[0].id)
    st_db = db_api.stack_get_all(self.ctx, show_nested=True)
    self.assertEqual(2, len(st_db))
    st_ids = [s.id for s in st_db]
    self.assertNotIn(stack3.id, st_ids)
    self.assertIn(stack1.id, st_ids)
    self.assertIn(stack2.id, st_ids)