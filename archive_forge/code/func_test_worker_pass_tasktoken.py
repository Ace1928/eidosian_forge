import boto.swf.layer2
from boto.swf.layer2 import Decider, ActivityWorker
from tests.unit import unittest
from mock import Mock
def test_worker_pass_tasktoken(self):
    task_token = 'worker_task_token'
    self.worker._swf.poll_for_activity_task.return_value = {'activityId': 'SomeActivity-1379020713', 'activityType': {'name': 'SomeActivity', 'version': '1.0'}, 'startedEventId': 6, 'taskToken': task_token, 'workflowExecution': {'runId': '12T026NzGK5c4eMti06N9O3GHFuTDaNyA+8LFtoDkAwfE=', 'workflowId': 'MyWorkflow-1.0-1379020705'}}
    self.worker.poll()
    self.worker.cancel(details='Cancelling!')
    self.worker.complete(result='Done!')
    self.worker.fail(reason='Failure!')
    self.worker.heartbeat()
    self.worker._swf.respond_activity_task_canceled.assert_called_with(task_token, 'Cancelling!')
    self.worker._swf.respond_activity_task_completed.assert_called_with(task_token, 'Done!')
    self.worker._swf.respond_activity_task_failed.assert_called_with(task_token, None, 'Failure!')
    self.worker._swf.record_activity_task_heartbeat.assert_called_with(task_token, None)