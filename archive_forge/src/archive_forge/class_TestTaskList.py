from osc_lib.cli import format_columns
from openstackclient.image.v2 import task
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
class TestTaskList(image_fakes.TestImagev2):
    tasks = image_fakes.create_tasks()
    columns = ('ID', 'Type', 'Status', 'Owner')
    datalist = [(task.id, task.type, task.status, task.owner_id) for task in tasks]

    def setUp(self):
        super().setUp()
        self.image_client.tasks.side_effect = [self.tasks, []]
        self.cmd = task.ListTask(self.app, None)

    def test_task_list_no_options(self):
        arglist = []
        verifylist = [('sort_key', None), ('sort_dir', None), ('limit', None), ('marker', None), ('type', None), ('status', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.tasks.assert_called_with()
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    def test_task_list_sort_key_option(self):
        arglist = ['--sort-key', 'created_at']
        verifylist = [('sort_key', 'created_at')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.image_client.tasks.assert_called_with(sort_key=parsed_args.sort_key)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    def test_task_list_sort_dir_option(self):
        arglist = ['--sort-dir', 'desc']
        verifylist = [('sort_dir', 'desc')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.image_client.tasks.assert_called_with(sort_dir=parsed_args.sort_dir)

    def test_task_list_pagination_options(self):
        arglist = ['--limit', '1', '--marker', self.tasks[0].id]
        verifylist = [('limit', 1), ('marker', self.tasks[0].id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.image_client.tasks.assert_called_with(limit=parsed_args.limit, marker=parsed_args.marker)

    def test_task_list_type_option(self):
        arglist = ['--type', self.tasks[0].type]
        verifylist = [('type', self.tasks[0].type)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.image_client.tasks.assert_called_with(type=self.tasks[0].type)

    def test_task_list_status_option(self):
        arglist = ['--status', self.tasks[0].status]
        verifylist = [('status', self.tasks[0].status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.image_client.tasks.assert_called_with(status=self.tasks[0].status)