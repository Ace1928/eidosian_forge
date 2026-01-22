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
def _deleted_stack_existance(self, ctx, stacks, resources, events, tmpl_files, existing, deleted):
    for s in existing:
        self.assertIsNotNone(db_api.stack_get(ctx, stacks[s].id, show_deleted=True))
        self.assertIsNotNone(db_api.raw_template_files_get(ctx, tmpl_files[s].files_id))
        self.assertIsNotNone(db_api.resource_get(ctx, resources[s].id))
        with db_api.context_manager.reader.using(ctx):
            self.assertIsNotNone(ctx.session.get(models.Event, events[s].id))
            self.assertIsNotNone(ctx.session.query(models.ResourcePropertiesData).filter_by(id=resources[s].rsrc_prop_data.id).first())
            self.assertIsNotNone(ctx.session.query(models.ResourcePropertiesData).filter_by(id=events[s].rsrc_prop_data.id).first())
    for s in deleted:
        self.assertIsNone(db_api.stack_get(ctx, stacks[s].id, show_deleted=True))
        rt_id = stacks[s].raw_template_id
        self.assertRaises(exception.NotFound, db_api.raw_template_get, ctx, rt_id)
        self.assertEqual({}, db_api.resource_get_all_by_stack(ctx, stacks[s].id))
        self.assertRaises(exception.NotFound, db_api.raw_template_files_get, ctx, tmpl_files[s].files_id)
        self.assertEqual([], db_api.event_get_all_by_stack(ctx, stacks[s].id))
        with db_api.context_manager.reader.using(ctx):
            self.assertIsNone(ctx.session.get(models.Event, events[s].id))
            self.assertIsNone(ctx.session.query(models.ResourcePropertiesData).filter_by(id=resources[s].rsrc_prop_data.id).first())
            self.assertIsNone(ctx.session.query(models.ResourcePropertiesData).filter_by(id=events[s].rsrc_prop_data.id).first())
        self.assertEqual([], db_api.event_get_all_by_stack(ctx, stacks[s].id))
        self.assertIsNone(db_api.user_creds_get(self.ctx, stacks[s].user_creds_id))