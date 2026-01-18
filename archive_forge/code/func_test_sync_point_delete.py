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
def test_sync_point_delete(self):
    for res in self.resources:
        sync_point_rsrc = create_sync_point(self.ctx, entity_id=str(res.id), stack_id=self.stack.id, traversal_id=self.stack.current_traversal)
        self.assertIsNotNone(sync_point_rsrc)
    sync_point_stack = create_sync_point(self.ctx, entity_id=self.stack.id, stack_id=self.stack.id, traversal_id=self.stack.current_traversal)
    self.assertIsNotNone(sync_point_stack)
    rows_deleted = db_api.sync_point_delete_all_by_stack_and_traversal(self.ctx, self.stack.id, self.stack.current_traversal)
    self.assertGreater(rows_deleted, 0)
    self.assertEqual(4, rows_deleted)
    for res in self.resources:
        ret_sync_point_rsrc = db_api.sync_point_get(self.ctx, str(res.id), self.stack.current_traversal, True)
        self.assertIsNone(ret_sync_point_rsrc)
    ret_sync_point_stack = db_api.sync_point_get(self.ctx, self.stack.id, self.stack.current_traversal, True)
    self.assertIsNone(ret_sync_point_stack)