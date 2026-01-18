from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
@mock.patch.object(notifier.LOG, 'warning')
def test_update_progress_handler_failure(self, mocked_warning):

    def progress_callback(*args, **kwargs):
        raise Exception('Woot!')
    a_task = ProgressTask()
    a_task.notifier.register(task.EVENT_UPDATE_PROGRESS, progress_callback)
    a_task.execute([0.5])
    self.assertEqual(1, mocked_warning.call_count)