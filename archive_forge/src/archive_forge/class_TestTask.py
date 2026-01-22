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
class TestTask(test_utils.BaseTestCase):

    def setUp(self):
        super(TestTask, self).setUp()
        self.task_factory = domain.TaskFactory()
        task_type = 'import'
        image_id = 'fake_image_id'
        user_id = 'fake_user'
        request_id = 'fake_request_id'
        owner = TENANT1
        task_ttl = CONF.task.task_time_to_live
        self.task = self.task_factory.new_task(task_type, owner, image_id, user_id, request_id, task_time_to_live=task_ttl)

    def test_task_invalid_status(self):
        task_id = str(uuid.uuid4())
        status = 'blah'
        self.assertRaises(exception.InvalidTaskStatus, domain.Task, task_id, task_type='import', status=status, owner=None, image_id='fake_image_id', user_id='fake_user', request_id='fake_request_id', expires_at=None, created_at=timeutils.utcnow(), updated_at=timeutils.utcnow(), task_input=None, message=None, result=None)

    def test_validate_status_transition_from_pending(self):
        self.task.begin_processing()
        self.assertEqual('processing', self.task.status)

    def test_validate_status_transition_from_processing_to_success(self):
        self.task.begin_processing()
        self.task.succeed('')
        self.assertEqual('success', self.task.status)

    def test_validate_status_transition_from_processing_to_failure(self):
        self.task.begin_processing()
        self.task.fail('')
        self.assertEqual('failure', self.task.status)

    def test_invalid_status_transitions_from_pending(self):
        self.assertRaises(exception.InvalidTaskStatusTransition, self.task.succeed, '')

    def test_invalid_status_transitions_from_success(self):
        self.task.begin_processing()
        self.task.succeed('')
        self.assertRaises(exception.InvalidTaskStatusTransition, self.task.begin_processing)
        self.assertRaises(exception.InvalidTaskStatusTransition, self.task.fail, '')

    def test_invalid_status_transitions_from_failure(self):
        self.task.begin_processing()
        self.task.fail('')
        self.assertRaises(exception.InvalidTaskStatusTransition, self.task.begin_processing)
        self.assertRaises(exception.InvalidTaskStatusTransition, self.task.succeed, '')

    def test_begin_processing(self):
        self.task.begin_processing()
        self.assertEqual('processing', self.task.status)

    @mock.patch.object(timeutils, 'utcnow')
    def test_succeed(self, mock_utcnow):
        mock_utcnow.return_value = datetime.datetime.utcnow()
        self.task.begin_processing()
        self.task.succeed('{"location": "file://home"}')
        self.assertEqual('success', self.task.status)
        self.assertEqual('{"location": "file://home"}', self.task.result)
        self.assertEqual(u'', self.task.message)
        expected = timeutils.utcnow() + datetime.timedelta(hours=CONF.task.task_time_to_live)
        self.assertEqual(expected, self.task.expires_at)

    @mock.patch.object(timeutils, 'utcnow')
    def test_fail(self, mock_utcnow):
        mock_utcnow.return_value = datetime.datetime.utcnow()
        self.task.begin_processing()
        self.task.fail('{"message": "connection failed"}')
        self.assertEqual('failure', self.task.status)
        self.assertEqual('{"message": "connection failed"}', self.task.message)
        self.assertIsNone(self.task.result)
        expected = timeutils.utcnow() + datetime.timedelta(hours=CONF.task.task_time_to_live)
        self.assertEqual(expected, self.task.expires_at)

    @mock.patch.object(glance.async_.TaskExecutor, 'begin_processing')
    def test_run(self, mock_begin_processing):
        executor = glance.async_.TaskExecutor(context=mock.ANY, task_repo=mock.ANY, image_repo=mock.ANY, image_factory=mock.ANY)
        self.task.run(executor)
        mock_begin_processing.assert_called_once_with(self.task.task_id)