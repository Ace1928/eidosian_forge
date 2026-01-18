from unittest import mock
import futurist
import glance_store
from oslo_config import cfg
from taskflow import engines
import glance.async_
from glance.async_ import taskflow_executor
from glance.common.scripts.image_import import main as image_import
from glance import domain
import glance.tests.utils as test_utils
def test_task_fail(self):
    with mock.patch.object(engines, 'load') as load_mock:
        engine = mock.Mock()
        load_mock.return_value = engine
        engine.run.side_effect = RuntimeError
        self.task_repo.get.return_value = self.task
        self.assertRaises(RuntimeError, self.executor.begin_processing, self.task.task_id)
    self.assertEqual('failure', self.task.status)
    self.task_repo.save.assert_called_with(self.task)