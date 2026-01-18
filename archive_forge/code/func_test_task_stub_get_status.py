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
def test_task_stub_get_status(self):
    status = 'pending'
    task = domain.TaskStub(self.task_id, self.task_type, status, self.owner, 'expires_at', 'created_at', 'updated_at', self.image_id, self.user_id, self.request_id)
    self.assertEqual(status, task.status)