from openstackclient.image.v2 import metadef_resource_types
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
class TestMetadefResourceTypeList(image_fakes.TestImagev2):
    resource_types = image_fakes.create_resource_types()
    columns = ['Name']
    datalist = [(resource_type.name,) for resource_type in resource_types]

    def setUp(self):
        super().setUp()
        self.image_client.metadef_resource_types.side_effect = [self.resource_types, []]
        self.cmd = metadef_resource_types.ListMetadefResourceTypes(self.app, None)

    def test_resource_type_list_no_options(self):
        arglist = []
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)