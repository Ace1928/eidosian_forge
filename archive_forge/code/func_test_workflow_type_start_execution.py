import boto.swf.layer2
from boto.swf.layer2 import ActivityType, WorkflowType, WorkflowExecution
from tests.unit import unittest
from mock import Mock, ANY
def test_workflow_type_start_execution(self):
    wf_type = WorkflowType(name='name', domain='test', version='1')
    run_id = '122aJcg6ic7MRAkjDRzLBsqU/R49qt5D0LPHycT/6ArN4='
    wf_type._swf.start_workflow_execution.return_value = {'runId': run_id}
    execution = wf_type.start(task_list='hello_world')
    self.assertIsInstance(execution, WorkflowExecution)
    self.assertEquals(wf_type.name, execution.name)
    self.assertEquals(wf_type.version, execution.version)
    self.assertEquals(run_id, execution.runId)