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
def test_resource_get_all_active_by_stack(self):
    values = [{'name': 'res1', 'action': rsrc.Resource.DELETE, 'status': rsrc.Resource.COMPLETE}, {'name': 'res2', 'action': rsrc.Resource.DELETE, 'status': rsrc.Resource.IN_PROGRESS}, {'name': 'res3', 'action': rsrc.Resource.UPDATE, 'status': rsrc.Resource.IN_PROGRESS}, {'name': 'res4', 'action': rsrc.Resource.UPDATE, 'status': rsrc.Resource.COMPLETE}, {'name': 'res5', 'action': rsrc.Resource.INIT, 'status': rsrc.Resource.COMPLETE}, {'name': 'res6'}]
    [create_resource(self.ctx, self.stack, **val) for val in values]
    resources = db_api.resource_get_all_active_by_stack(self.ctx, self.stack.id)
    self.assertEqual(5, len(resources))
    for rsrc_id, res in resources.items():
        self.assertIn(res.name, ['res2', 'res3', 'res4', 'res5', 'res6'])