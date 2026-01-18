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
def test_sync_point_create_get(self):
    for res in self.resources:
        sync_point_rsrc = create_sync_point(self.ctx, entity_id=str(res.id), stack_id=self.stack.id, traversal_id=self.stack.current_traversal)
        ret_sync_point_rsrc = db_api.sync_point_get(self.ctx, sync_point_rsrc.entity_id, sync_point_rsrc.traversal_id, sync_point_rsrc.is_update)
        self.assertIsNotNone(ret_sync_point_rsrc)
        self.assertEqual(sync_point_rsrc.entity_id, ret_sync_point_rsrc.entity_id)
        self.assertEqual(sync_point_rsrc.traversal_id, ret_sync_point_rsrc.traversal_id)
        self.assertEqual(sync_point_rsrc.is_update, ret_sync_point_rsrc.is_update)
        self.assertEqual(sync_point_rsrc.atomic_key, ret_sync_point_rsrc.atomic_key)
        self.assertEqual(sync_point_rsrc.stack_id, ret_sync_point_rsrc.stack_id)
        self.assertEqual(sync_point_rsrc.input_data, ret_sync_point_rsrc.input_data)
    sync_point_stack = create_sync_point(self.ctx, entity_id=self.stack.id, stack_id=self.stack.id, traversal_id=self.stack.current_traversal)
    ret_sync_point_stack = db_api.sync_point_get(self.ctx, sync_point_stack.entity_id, sync_point_stack.traversal_id, sync_point_stack.is_update)
    self.assertIsNotNone(ret_sync_point_stack)
    self.assertEqual(sync_point_stack.entity_id, ret_sync_point_stack.entity_id)
    self.assertEqual(sync_point_stack.traversal_id, ret_sync_point_stack.traversal_id)
    self.assertEqual(sync_point_stack.is_update, ret_sync_point_stack.is_update)
    self.assertEqual(sync_point_stack.atomic_key, ret_sync_point_stack.atomic_key)
    self.assertEqual(sync_point_stack.stack_id, ret_sync_point_stack.stack_id)
    self.assertEqual(sync_point_stack.input_data, ret_sync_point_stack.input_data)