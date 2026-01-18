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
def test_user_creds_delete_retries(self):
    mock_delete = self.patchobject(session.Session, 'delete')
    mock_delete.side_effect = [exc.StaleDataError, exc.StaleDataError, None]
    user_creds = create_user_creds(self.ctx)
    self.assertIsNotNone(user_creds['id'])
    self.assertIsNone(db_api.user_creds_delete(self.ctx, user_creds['id']))
    self.assertEqual(3, mock_delete.call_count)
    mock_delete.side_effect = [exc.UnmappedError]
    self.assertRaises(exc.UnmappedError, db_api.user_creds_delete, self.ctx, user_creds['id'])
    self.assertEqual(4, mock_delete.call_count)