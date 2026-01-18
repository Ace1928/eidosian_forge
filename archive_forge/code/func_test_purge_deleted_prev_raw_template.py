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
def test_purge_deleted_prev_raw_template(self):
    now = timeutils.utcnow()
    templates = [create_raw_template(self.ctx) for i in range(2)]
    stacks = [create_stack(self.ctx, templates[0], create_user_creds(self.ctx), deleted_at=now - datetime.timedelta(seconds=10), prev_raw_template=templates[1])]
    db_api.purge_deleted(age=3600, granularity='seconds')
    ctx = utils.dummy_context(is_admin=True)
    self.assertIsNotNone(db_api.stack_get(ctx, stacks[0].id, show_deleted=True))
    self.assertIsNotNone(db_api.raw_template_get(ctx, templates[1].id))
    stacks = [create_stack(self.ctx, templates[0], create_user_creds(self.ctx), deleted_at=now - datetime.timedelta(seconds=10), prev_raw_template=templates[1], tenant=UUID1)]
    db_api.purge_deleted(age=3600, granularity='seconds', project_id=UUID1)
    self.assertIsNotNone(db_api.stack_get(ctx, stacks[0].id, show_deleted=True))
    self.assertIsNotNone(db_api.raw_template_get(ctx, templates[1].id))
    db_api.purge_deleted(age=0, granularity='seconds', project_id=UUID2)
    self.assertIsNotNone(db_api.stack_get(ctx, stacks[0].id, show_deleted=True))
    self.assertIsNotNone(db_api.raw_template_get(ctx, templates[1].id))