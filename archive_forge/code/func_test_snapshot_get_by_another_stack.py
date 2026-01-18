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
def test_snapshot_get_by_another_stack(self):
    template = create_raw_template(self.ctx)
    user_creds = create_user_creds(self.ctx)
    stack = create_stack(self.ctx, template, user_creds)
    stack1 = create_stack(self.ctx, template, user_creds)
    values = {'tenant': self.ctx.tenant_id, 'status': 'IN_PROGRESS', 'stack_id': stack.id}
    snapshot = db_api.snapshot_create(self.ctx, values)
    self.assertIsNotNone(snapshot)
    snapshot_id = snapshot.id
    self.assertRaises(exception.SnapshotNotFound, db_api.snapshot_get_by_stack, self.ctx, snapshot_id, stack1)