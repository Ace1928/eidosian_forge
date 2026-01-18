from openstackclient.image.v2 import metadef_resource_types
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def test_resource_type_list_no_options(self):
    arglist = []
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, data)