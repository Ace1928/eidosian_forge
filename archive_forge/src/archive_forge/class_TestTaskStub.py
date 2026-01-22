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
class TestTaskStub(test_utils.BaseTestCase):

    def setUp(self):
        super(TestTaskStub, self).setUp()
        self.task_id = str(uuid.uuid4())
        self.task_type = 'import'
        self.owner = TENANT1
        self.task_ttl = CONF.task.task_time_to_live
        self.image_id = 'fake_image_id'
        self.user_id = 'fake_user'
        self.request_id = 'fake_request_id'

    def test_task_stub_init(self):
        self.task_factory = domain.TaskFactory()
        task = domain.TaskStub(self.task_id, self.task_type, 'status', self.owner, 'expires_at', 'created_at', 'updated_at', self.image_id, self.user_id, self.request_id)
        self.assertEqual(self.task_id, task.task_id)
        self.assertEqual(self.task_type, task.type)
        self.assertEqual(self.owner, task.owner)
        self.assertEqual('status', task.status)
        self.assertEqual('expires_at', task.expires_at)
        self.assertEqual('created_at', task.created_at)
        self.assertEqual('updated_at', task.updated_at)
        self.assertEqual(self.image_id, task.image_id)
        self.assertEqual(self.user_id, task.user_id)
        self.assertEqual(self.request_id, task.request_id)

    def test_task_stub_get_status(self):
        status = 'pending'
        task = domain.TaskStub(self.task_id, self.task_type, status, self.owner, 'expires_at', 'created_at', 'updated_at', self.image_id, self.user_id, self.request_id)
        self.assertEqual(status, task.status)