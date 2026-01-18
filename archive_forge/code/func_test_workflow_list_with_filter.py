import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_workflow_list_with_filter(self):
    self.workflow_create(self.wf_def)
    workflows = self.parser.listing(self.mistral('workflow-list'))
    self.assertTableStruct(workflows, ['ID', 'Name', 'Tags', 'Input', 'Scope', 'Created at', 'Updated at'])
    self.assertGreaterEqual(len(workflows), 2)
    workflows = self.parser.listing(self.mistral('workflow-list', params='--filter name=eq:wf1'))
    self.assertTableStruct(workflows, ['ID', 'Name', 'Tags', 'Input', 'Scope', 'Created at', 'Updated at'])
    self.assertEqual(1, len(workflows))
    self.assertEqual('wf1', workflows[0]['Name'])