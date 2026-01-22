import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
class TaskCLITests(base_v2.MistralClientTestBase):
    """Test suite checks commands to work with tasks."""

    def setUp(self):
        super(TaskCLITests, self).setUp()
        wfs = self.workflow_create(self.wf_def)
        self.direct_wf = wfs[0]
        self.reverse_wf = wfs[1]
        self.create_file('input', '{\n    "farewell": "Bye"\n}\n')
        self.create_file('task_name', '{\n    "task_name": "goodbye"\n}\n')

    def test_task_get(self):
        wf_ex = self.execution_create(self.direct_wf['Name'])
        wf_ex_id = self.get_field_value(wf_ex, 'ID')
        tasks = self.mistral_admin('task-list', params=wf_ex_id)
        created_task_id = tasks[-1]['ID']
        fetched_task = self.mistral_admin('task-get', params=created_task_id)
        fetched_task_id = self.get_field_value(fetched_task, 'ID')
        fetched_task_wf_namespace = self.get_field_value(fetched_task, 'Workflow namespace')
        task_execution_id = self.get_field_value(fetched_task, 'Workflow Execution ID')
        self.assertEqual(created_task_id, fetched_task_id)
        self.assertEqual('', fetched_task_wf_namespace)
        self.assertEqual(wf_ex_id, task_execution_id)

    def test_task_get_list_within_namespace(self):
        namespace = 'aaa'
        self.workflow_create(self.wf_def, namespace=namespace)
        wf_ex = self.execution_create(self.direct_wf['Name'] + ' --namespace ' + namespace)
        wf_ex_id = self.get_field_value(wf_ex, 'ID')
        tasks = self.mistral_admin('task-list', params=wf_ex_id)
        created_task_id = tasks[-1]['ID']
        created_wf_namespace = tasks[-1]['Workflow namespace']
        fetched_task = self.mistral_admin('task-get', params=created_task_id)
        fetched_task_id = self.get_field_value(fetched_task, 'ID')
        fetched_task_wf_namespace = self.get_field_value(fetched_task, 'Workflow namespace')
        task_execution_id = self.get_field_value(fetched_task, 'Workflow Execution ID')
        self.assertEqual(created_task_id, fetched_task_id)
        self.assertEqual(namespace, created_wf_namespace)
        self.assertEqual(created_wf_namespace, fetched_task_wf_namespace)
        self.assertEqual(wf_ex_id, task_execution_id)

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

    def test_task_list_with_limit(self):
        wf_exec = self.execution_create('%s input task_name' % self.reverse_wf['Name'])
        exec_id = self.get_field_value(wf_exec, 'ID')
        self.assertTrue(self.wait_execution_success(exec_id))
        tasks = self.parser.listing(self.mistral('task-list'))
        tasks = self.parser.listing(self.mistral('task-list', params='--limit 1'))
        self.assertEqual(1, len(tasks))