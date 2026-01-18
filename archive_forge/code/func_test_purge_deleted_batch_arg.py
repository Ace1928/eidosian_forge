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
def test_purge_deleted_batch_arg(self):
    now = timeutils.utcnow()
    delta = datetime.timedelta(seconds=3600)
    deleted = now - delta
    for i in range(7):
        create_stack(self.ctx, self.template, self.user_creds, deleted_at=deleted)
    with mock.patch('heat.db.api._purge_stacks') as mock_ps:
        db_api.purge_deleted(age=0, batch_size=2)
        self.assertEqual(4, mock_ps.call_count)