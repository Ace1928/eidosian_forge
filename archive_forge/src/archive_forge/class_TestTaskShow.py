from osc_lib.cli import format_columns
from openstackclient.image.v2 import task
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
class TestTaskShow(image_fakes.TestImagev2):
    task = image_fakes.create_one_task()
    columns = ('created_at', 'expires_at', 'id', 'input', 'message', 'owner_id', 'properties', 'result', 'status', 'type', 'updated_at')
    data = (task.created_at, task.expires_at, task.id, task.input, task.message, task.owner_id, format_columns.DictColumn({}), task.result, task.status, task.type, task.updated_at)

    def setUp(self):
        super().setUp()
        self.image_client.get_task.return_value = self.task
        self.cmd = task.ShowTask(self.app, None)

    def test_task_show(self):
        arglist = [self.task.id]
        verifylist = [('task', self.task.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.get_task.assert_called_with(self.task.id)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)