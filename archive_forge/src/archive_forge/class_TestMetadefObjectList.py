from openstackclient.image.v2 import metadef_objects
from openstackclient.tests.unit.image.v2 import fakes
class TestMetadefObjectList(fakes.TestImagev2):
    _metadef_namespace = fakes.create_one_metadef_namespace()
    _metadef_objects = [fakes.create_one_metadef_object()]
    columns = ['name', 'description']
    datalist = []

    def setUp(self):
        super().setUp()
        self.image_client.metadef_objects.side_effect = [self._metadef_objects, []]
        self.image_client.metadef_objects.return_value = iter(self._metadef_objects)
        self.cmd = metadef_objects.ListMetadefObjects(self.app, None)
        self.datalist = self._metadef_objects

    def test_metadef_objects_list(self):
        arglist = [self._metadef_namespace.namespace]
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(getattr(self.datalist[0], 'name'), next(data)[0])