import datetime
from unittest import mock
import uuid
from oslo_config import cfg
import oslo_utils.importutils
import glance.async_
from glance.async_ import taskflow_executor
from glance.common import exception
from glance.common import timeutils
from glance import domain
import glance.tests.utils as test_utils
def test_task_invalid_status(self):
    task_id = str(uuid.uuid4())
    status = 'blah'
    self.assertRaises(exception.InvalidTaskStatus, domain.Task, task_id, task_type='import', status=status, owner=None, image_id='fake_image_id', user_id='fake_user', request_id='fake_request_id', expires_at=None, created_at=timeutils.utcnow(), updated_at=timeutils.utcnow(), task_input=None, message=None, result=None)