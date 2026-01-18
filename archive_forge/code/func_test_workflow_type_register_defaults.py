import boto.swf.layer2
from boto.swf.layer2 import ActivityType, WorkflowType, WorkflowExecution
from tests.unit import unittest
from mock import Mock, ANY
def test_workflow_type_register_defaults(self):
    wf_type = WorkflowType(name='name', domain='test', version='1')
    wf_type.register()
    wf_type._swf.register_workflow_type.assert_called_with('test', 'name', '1', default_execution_start_to_close_timeout=ANY, default_task_start_to_close_timeout=ANY, default_child_policy=ANY)