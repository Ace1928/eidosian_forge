from openstackclient.image.v2 import metadef_namespaces
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
class TestMetadefNamespaceDelete(image_fakes.TestImagev2):
    _metadef_namespace = image_fakes.create_one_metadef_namespace()

    def setUp(self):
        super().setUp()
        self.image_client.delete_metadef_namespace.return_value = self._metadef_namespace
        self.cmd = metadef_namespaces.DeleteMetadefNamespace(self.app, None)
        self.datalist = self._metadef_namespace

    def test_namespace_create(self):
        arglist = [self._metadef_namespace.namespace]
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)