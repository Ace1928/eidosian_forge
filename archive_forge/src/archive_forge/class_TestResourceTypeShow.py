from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource_type
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resource_types
class TestResourceTypeShow(TestResourceType):

    def setUp(self):
        super(TestResourceTypeShow, self).setUp()
        self.cmd = resource_type.ResourceTypeShow(self.app, None)
        self.mock_client.resource_types.get.return_value = {}
        self.mock_client.resource_types.generate_template.return_value = {}

    def test_resourcetype_show(self):
        arglist = ['OS::Heat::None']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_client.resource_types.get.assert_called_once_with('OS::Heat::None', False)

    def test_resourcetype_show_json(self):
        arglist = ['OS::Heat::None', '--format', 'json']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_client.resource_types.get.assert_called_once_with('OS::Heat::None', False)

    def test_resourcetype_show_error_get(self):
        arglist = ['OS::Heat::None']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.mock_client.resource_types.get.side_effect = heat_exc.HTTPNotFound
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)

    def test_resourcetype_show_error_template(self):
        arglist = ['OS::Heat::None', '--template-type', 'hot']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.mock_client.resource_types.generate_template.side_effect = heat_exc.HTTPNotFound
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)

    def test_resourcetype_show_template_hot(self):
        arglist = ['OS::Heat::None', '--template-type', 'Hot']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_client.resource_types.generate_template.assert_called_with(**{'resource_type': 'OS::Heat::None', 'template_type': 'hot'})

    def test_resourcetype_show_template_cfn(self):
        arglist = ['OS::Heat::None', '--template-type', 'cfn']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_client.resource_types.generate_template.assert_called_with(**{'resource_type': 'OS::Heat::None', 'template_type': 'cfn'})

    def test_resourcetype_show_template_cfn_yaml(self):
        arglist = ['OS::Heat::None', '--template-type', 'Cfn', '--format', 'yaml']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_client.resource_types.generate_template.assert_called_with(**{'resource_type': 'OS::Heat::None', 'template_type': 'cfn'})

    def test_resourcetype_show_invalid_template_type(self):
        arglist = ['OS::Heat::None', '--template-type', 'abc']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)

    def test_resourcetype_show_with_description(self):
        arglist = ['OS::Heat::None', '--long']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.mock_client.resource_types.get.assert_called_with('OS::Heat::None', True)

    def test_resourcetype_show_long_and_template_type_error(self):
        arglist = ['OS::Heat::None', '--template-type', 'cfn', '--long']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)