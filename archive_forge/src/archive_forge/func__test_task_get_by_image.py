import copy
import datetime
import functools
from unittest import mock
import uuid
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import utils as sqlalchemyutils
from sqlalchemy import sql
from glance.common import exception
from glance.common import timeutils
from glance import context
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import models
from glance.tests import functional
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def _test_task_get_by_image(self, expired=False, deleted=False, other_owner=False):
    expires_at = timeutils.utcnow()
    if expired is False:
        expires_at += datetime.timedelta(hours=1)
    elif expired is None:
        expires_at = None
    image_id = str(uuid.uuid4())
    fixture = {'owner': other_owner and 'notme!' or self.context.owner, 'type': 'import', 'status': 'pending', 'input': '{"loc": "fake"}', 'result': "{'image_id': %s}" % image_id, 'message': 'blah', 'expires_at': expires_at, 'image_id': image_id, 'user_id': 'me', 'request_id': 'reqid'}
    new_task = self.db_api.task_create(self.adm_context, fixture)
    if deleted:
        self.db_api.task_delete(self.context, new_task['id'])
    return (new_task['id'], self.db_api.tasks_get_by_image(self.context, image_id))