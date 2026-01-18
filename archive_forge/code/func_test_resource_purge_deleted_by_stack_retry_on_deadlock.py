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
@mock.patch.object(time, 'sleep')
def test_resource_purge_deleted_by_stack_retry_on_deadlock(self, m_sleep):
    val = {'name': 'res1', 'action': rsrc.Resource.DELETE, 'status': rsrc.Resource.COMPLETE}
    create_resource(self.ctx, self.stack, **val)
    with mock.patch('sqlalchemy.orm.query.Query.delete', side_effect=db_exception.DBDeadlock) as mock_delete:
        self.assertRaises(db_exception.DBDeadlock, db_api.resource_purge_deleted, self.ctx, self.stack.id)
        self.assertEqual(21, mock_delete.call_count)