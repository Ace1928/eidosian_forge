from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.types import notifier
def test_update_progress_within_bounds(self):
    values = [0.0, 0.5, 1.0]
    result = []

    def progress_callback(event_type, details):
        result.append(details.pop('progress'))
    a_task = ProgressTask()
    a_task.notifier.register(task.EVENT_UPDATE_PROGRESS, progress_callback)
    a_task.execute(values)
    self.assertEqual(values, result)