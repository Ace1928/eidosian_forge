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
def test_resource_create_replacement_updated(self):
    orig = create_resource(self.ctx, self.stack)
    other_ctx = utils.dummy_context()
    tmpl_id = create_raw_template(self.ctx).id
    db_api.resource_update_and_save(other_ctx, orig.id, {'atomic_key': 2})
    self.assertRaises(exception.UpdateInProgress, db_api.resource_create_replacement, self.ctx, orig.id, {'name': orig.name, 'replaces': orig.id, 'stack_id': orig.stack_id, 'current_template_id': tmpl_id}, 1, None)