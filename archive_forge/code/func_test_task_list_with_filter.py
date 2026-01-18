import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_task_list_with_filter(self):
    wf_exec = self.execution_create('%s input task_name' % self.reverse_wf['Name'])
    exec_id = self.get_field_value(wf_exec, 'ID')
    self.assertTrue(self.wait_execution_success(exec_id))
    tasks = self.parser.listing(self.mistral('task-list'))
    self.assertTableStruct(tasks, ['ID', 'Name', 'Workflow name', 'Workflow Execution ID', 'State'])
    self.assertEqual(2, len(tasks))
    tasks = self.parser.listing(self.mistral('task-list', params='--filter name=goodbye'))
    self.assertTableStruct(tasks, ['ID', 'Name', 'Workflow name', 'Workflow Execution ID', 'State'])
    self.assertEqual(1, len(tasks))
    self.assertEqual('goodbye', tasks[0]['Name'])