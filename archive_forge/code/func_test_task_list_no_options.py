from osc_lib.cli import format_columns
from openstackclient.image.v2 import task
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_task_list_no_options(self):
    arglist = []
    verifylist = [('sort_key', None), ('sort_dir', None), ('limit', None), ('marker', None), ('type', None), ('status', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.image_client.tasks.assert_called_with()
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, data)