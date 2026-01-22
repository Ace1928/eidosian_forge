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
class TestTaskExecutorFactory(test_utils.BaseTestCase):

    def setUp(self):
        super(TestTaskExecutorFactory, self).setUp()
        self.task_repo = mock.Mock()
        self.image_repo = mock.Mock()
        self.image_factory = mock.Mock()

    def test_init(self):
        task_executor_factory = domain.TaskExecutorFactory(self.task_repo, self.image_repo, self.image_factory)
        self.assertEqual(self.task_repo, task_executor_factory.task_repo)

    def test_new_task_executor(self):
        task_executor_factory = domain.TaskExecutorFactory(self.task_repo, self.image_repo, self.image_factory)
        context = mock.Mock()
        with mock.patch.object(oslo_utils.importutils, 'import_class') as mock_import_class:
            mock_executor = mock.Mock()
            mock_import_class.return_value = mock_executor
            task_executor_factory.new_task_executor(context)
        mock_executor.assert_called_once_with(context, self.task_repo, self.image_repo, self.image_factory, admin_repo=None)

    def test_new_task_executor_with_admin(self):
        admin_repo = mock.MagicMock()
        task_executor_factory = domain.TaskExecutorFactory(self.task_repo, self.image_repo, self.image_factory, admin_repo=admin_repo)
        context = mock.Mock()
        with mock.patch.object(oslo_utils.importutils, 'import_class') as mock_import_class:
            mock_executor = mock.Mock()
            mock_import_class.return_value = mock_executor
            task_executor_factory.new_task_executor(context)
        mock_executor.assert_called_once_with(context, self.task_repo, self.image_repo, self.image_factory, admin_repo=admin_repo)

    def test_new_task_executor_error(self):
        task_executor_factory = domain.TaskExecutorFactory(self.task_repo, self.image_repo, self.image_factory)
        context = mock.Mock()
        with mock.patch.object(oslo_utils.importutils, 'import_class') as mock_import_class:
            mock_import_class.side_effect = ImportError
            self.assertRaises(ImportError, task_executor_factory.new_task_executor, context)

    def test_new_task_eventlet_backwards_compatibility(self):
        context = mock.MagicMock()
        self.config(task_executor='eventlet', group='task')
        task_executor_factory = domain.TaskExecutorFactory(self.task_repo, self.image_repo, self.image_factory)
        te_evnt = task_executor_factory.new_task_executor(context)
        self.assertIsInstance(te_evnt, taskflow_executor.TaskExecutor)