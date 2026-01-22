from openstack import exceptions as sdk_exceptions
from osc_lib import exceptions
from openstackclient.image.v2 import metadef_properties
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestMetadefPropertyCreate(image_fakes.TestImagev2):
    _metadef_namespace = image_fakes.create_one_metadef_namespace()
    _metadef_property = image_fakes.create_one_metadef_property()
    expected_columns = ('name', 'title', 'type')
    expected_data = (_metadef_property.name, _metadef_property.title, _metadef_property.type)

    def setUp(self):
        super().setUp()
        self.image_client.create_metadef_property.return_value = self._metadef_property
        self.cmd = metadef_properties.CreateMetadefProperty(self.app, None)

    def test_metadef_property_create_missing_arguments(self):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_metadef_property_create(self):
        arglist = ['--name', 'cpu_cores', '--schema', '{}', '--title', 'vCPU Cores', '--type', 'integer', self._metadef_namespace.namespace]
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)

    def test_metadef_property_create_invalid_schema(self):
        arglist = ['--name', 'cpu_cores', '--schema', '{invalid}', '--title', 'vCPU Cores', '--type', 'integer', self._metadef_namespace.namespace]
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)