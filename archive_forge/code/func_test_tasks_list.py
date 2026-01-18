import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_tasks_list(self):
    tasks = self.parser.listing(self.mistral('task-list'))
    self.assertTableStruct(tasks, ['ID', 'Name', 'Workflow name', 'Workflow Execution ID', 'State'])