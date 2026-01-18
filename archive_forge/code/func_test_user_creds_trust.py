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
def test_user_creds_trust(self):
    self.ctx.username = None
    self.ctx.password = None
    self.ctx.trust_id = 'atrust123'
    self.ctx.trustor_user_id = 'atrustor123'
    self.ctx.project_id = 'atenant123'
    self.ctx.project_name = 'atenant'
    self.ctx.auth_url = 'anauthurl'
    self.ctx.region_name = 'aregion'
    db_creds = db_api.user_creds_create(self.ctx)
    load_creds = db_api.user_creds_get(self.ctx, db_creds['id'])
    self.assertIsNone(load_creds.get('username'))
    self.assertIsNone(load_creds.get('password'))
    self.assertIsNotNone(load_creds.get('created_at'))
    self.assertIsNone(load_creds.get('updated_at'))
    self.assertEqual('anauthurl', load_creds.get('auth_url'))
    self.assertEqual('aregion', load_creds.get('region_name'))
    self.assertEqual('atenant123', load_creds.get('tenant_id'))
    self.assertEqual('atenant', load_creds.get('tenant'))
    self.assertEqual('atrust123', load_creds.get('trust_id'))
    self.assertEqual('atrustor123', load_creds.get('trustor_user_id'))