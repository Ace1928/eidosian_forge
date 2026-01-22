from openstack import exceptions as sdk_exceptions
from osc_lib import exceptions
from openstackclient.image.v2 import metadef_properties
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestMetadefPropertyShow(image_fakes.TestImagev2):
    _metadef_property = image_fakes.create_one_metadef_property()
    expected_columns = ('name', 'title', 'type')
    expected_data = (_metadef_property.name, _metadef_property.title, _metadef_property.type)

    def setUp(self):
        super().setUp()
        self.image_client.get_metadef_property.return_value = self._metadef_property
        self.cmd = metadef_properties.ShowMetadefProperty(self.app, None)

    def test_metadef_property_show(self):
        arglist = ['my-namespace', 'my-property']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.expected_columns, columns)
        self.assertEqual(self.expected_data, data)