from unittest import mock
import futurist
import glance_store as store
from oslo_config import cfg
from taskflow.patterns import linear_flow
import glance.async_
from glance.async_.flows import api_image_import
import glance.tests.utils as test_utils
class TestTaskExecutor(test_utils.BaseTestCase):

    def setUp(self):
        super(TestTaskExecutor, self).setUp()
        self.context = mock.Mock()
        self.task_repo = mock.Mock()
        self.image_repo = mock.Mock()
        self.image_factory = mock.Mock()
        self.executor = glance.async_.TaskExecutor(self.context, self.task_repo, self.image_repo, self.image_factory)

    def test_begin_processing(self):
        task_id = mock.ANY
        task_type = mock.ANY
        task = mock.Mock()
        with mock.patch.object(glance.async_.TaskExecutor, '_run') as mock_run:
            self.task_repo.get.return_value = task
            self.executor.begin_processing(task_id)
        mock_run.assert_called_once_with(task_id, task_type)

    def test_with_admin_repo(self):
        admin_repo = mock.MagicMock()
        executor = glance.async_.TaskExecutor(self.context, self.task_repo, self.image_repo, self.image_factory, admin_repo=admin_repo)
        self.assertEqual(admin_repo, executor.admin_repo)