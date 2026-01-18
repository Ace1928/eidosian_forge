from osc_lib.cli import format_columns
from openstackclient.image.v2 import task
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_task_list_pagination_options(self):
    arglist = ['--limit', '1', '--marker', self.tasks[0].id]
    verifylist = [('limit', 1), ('marker', self.tasks[0].id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.image_client.tasks.assert_called_with(limit=parsed_args.limit, marker=parsed_args.marker)