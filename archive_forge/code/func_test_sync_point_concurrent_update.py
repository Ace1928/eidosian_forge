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
def test_sync_point_concurrent_update(self):
    sync_point = create_sync_point(self.ctx, entity_id=str(self.resources[0].id), stack_id=self.stack.id, traversal_id=self.stack.current_traversal)
    self.assertEqual({}, sync_point.input_data)
    self.assertEqual(0, sync_point.atomic_key)
    rows_updated = db_api.sync_point_update_input_data(self.ctx, sync_point.entity_id, sync_point.traversal_id, sync_point.is_update, 0, {'input_data': '{key: value}'})
    self.assertEqual(1, rows_updated)
    rows_updated = db_api.sync_point_update_input_data(self.ctx, sync_point.entity_id, sync_point.traversal_id, sync_point.is_update, 0, {'input_data': '{key: value}'})
    self.assertEqual(0, rows_updated)