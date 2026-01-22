from openstackclient.image.v2 import metadef_namespaces
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
class TestMetadefNamespaceList(image_fakes.TestImagev2):
    _metadef_namespace = [image_fakes.create_one_metadef_namespace()]
    columns = ['namespace']
    datalist = []

    def setUp(self):
        super().setUp()
        self.image_client.metadef_namespaces.side_effect = [self._metadef_namespace, []]
        self.image_client.metadef_namespaces.return_value = iter(self._metadef_namespace)
        self.cmd = metadef_namespaces.ListMetadefNamespace(self.app, None)
        self.datalist = self._metadef_namespace

    def test_namespace_list_no_options(self):
        arglist = []
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(getattr(self.datalist[0], 'namespace'), next(data)[0])