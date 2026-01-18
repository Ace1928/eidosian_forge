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
def test_software_config_get(self):
    self.assertRaises(exception.NotFound, db_api.software_config_get, self.ctx, str(uuid.uuid4()))
    conf = '#!/bin/bash\necho "$bar and $foo"\n'
    config = {'inputs': [{'name': 'foo'}, {'name': 'bar'}], 'outputs': [{'name': 'result'}], 'config': conf, 'options': {}}
    tenant_id = self.ctx.tenant_id
    values = {'name': 'config_mysql', 'tenant': tenant_id, 'group': 'Heat::Shell', 'config': config}
    config = db_api.software_config_create(self.ctx, values)
    config_id = config.id
    config = db_api.software_config_get(self.ctx, config_id)
    self.assertIsNotNone(config)
    self.assertEqual('config_mysql', config.name)
    self.assertEqual(tenant_id, config.tenant)
    self.assertEqual('Heat::Shell', config.group)
    self.assertEqual(conf, config.config['config'])
    self.ctx.project_id = None
    self.assertRaises(exception.NotFound, db_api.software_config_get, self.ctx, config_id)
    admin_ctx = utils.dummy_context(is_admin=True, tenant_id='admin_tenant')
    config = db_api.software_config_get(admin_ctx, config_id)
    self.assertIsNotNone(config)