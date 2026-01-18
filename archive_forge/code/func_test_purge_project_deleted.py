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
def test_purge_project_deleted(self):
    now = timeutils.utcnow()
    delta = datetime.timedelta(seconds=3600 * 7)
    deleted = [now - delta * i for i in range(1, 6)]
    tmpl_files = [template_files.TemplateFiles({'foo': 'file contents %d' % i}) for i in range(5)]
    [tmpl_file.store(self.ctx) for tmpl_file in tmpl_files]
    templates = [create_raw_template(self.ctx, files_id=tmpl_files[i].files_id) for i in range(5)]
    values = [{'tenant': UUID1}, {'tenant': UUID1}, {'tenant': UUID1}, {'tenant': UUID2}, {'tenant': UUID2}]
    creds = [create_user_creds(self.ctx) for i in range(5)]
    stacks = [create_stack(self.ctx, templates[i], creds[i], deleted_at=deleted[i], **values[i]) for i in range(5)]
    resources = [create_resource(self.ctx, stacks[i]) for i in range(5)]
    events = [create_event(self.ctx, stack_id=stacks[i].id) for i in range(5)]
    db_api.purge_deleted(age=1, granularity='days', project_id=UUID1)
    admin_ctx = utils.dummy_context(is_admin=True)
    self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (0, 1, 2, 3, 4), ())
    db_api.purge_deleted(age=22, granularity='hours', project_id=UUID1)
    self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (0, 1, 2, 3, 4), ())
    db_api.purge_deleted(age=1100, granularity='minutes', project_id=UUID1)
    self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (0, 1, 3, 4), (2,))
    db_api.purge_deleted(age=30, granularity='hours', project_id=UUID2)
    self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (0, 1, 3), (2, 4))
    db_api.purge_deleted(age=3600, granularity='seconds', project_id=UUID1)
    self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (3,), (0, 1, 2, 4))
    db_api.purge_deleted(age=3600, granularity='seconds', project_id=UUID2)
    self._deleted_stack_existance(admin_ctx, stacks, resources, events, tmpl_files, (), (0, 1, 2, 3, 4))