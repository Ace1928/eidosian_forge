from unittest import mock
from openstack.compute.v2 import flavor as _flavor
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import flavor
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestFlavorSet(TestFlavor):
    flavor = compute_fakes.create_one_flavor(attrs={'os-flavor-access:is_public': False})
    project = identity_fakes.FakeProject.create_one_project()

    def setUp(self):
        super(TestFlavorSet, self).setUp()
        self.compute_sdk_client.find_flavor.return_value = self.flavor
        self.projects_mock.get.return_value = self.project
        self.cmd = flavor.SetFlavor(self.app, None)

    def test_flavor_set_property(self):
        arglist = ['--property', 'FOO="B A R"', 'baremetal']
        verifylist = [('properties', {'FOO': '"B A R"'}), ('flavor', 'baremetal')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_flavor.assert_called_with(parsed_args.flavor, get_extra_specs=True, ignore_missing=False)
        self.compute_sdk_client.create_flavor_extra_specs.assert_called_with(self.flavor.id, {'FOO': '"B A R"'})
        self.assertIsNone(result)

    def test_flavor_set_no_property(self):
        arglist = ['--no-property', 'baremetal']
        verifylist = [('no_property', True), ('flavor', 'baremetal')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_flavor.assert_called_with(parsed_args.flavor, get_extra_specs=True, ignore_missing=False)
        self.compute_sdk_client.delete_flavor_extra_specs_property.assert_called_with(self.flavor.id, 'property')
        self.assertIsNone(result)

    def test_flavor_set_project(self):
        arglist = ['--project', self.project.id, self.flavor.id]
        verifylist = [('project', self.project.id), ('flavor', self.flavor.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_flavor.assert_called_with(parsed_args.flavor, get_extra_specs=True, ignore_missing=False)
        self.compute_sdk_client.flavor_add_tenant_access.assert_called_with(self.flavor.id, self.project.id)
        self.compute_sdk_client.create_flavor_extra_specs.assert_not_called()
        self.assertIsNone(result)

    def test_flavor_set_no_project(self):
        arglist = ['--project', self.flavor.id]
        verifylist = [('project', None), ('flavor', self.flavor.id)]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_flavor_set_no_flavor(self):
        arglist = ['--project', self.project.id]
        verifylist = [('project', self.project.id)]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_flavor_set_with_unexist_flavor(self):
        self.compute_sdk_client.find_flavor.side_effect = [sdk_exceptions.ResourceNotFound()]
        arglist = ['--project', self.project.id, 'unexist_flavor']
        verifylist = [('project', self.project.id), ('flavor', 'unexist_flavor')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_flavor_set_nothing(self):
        arglist = [self.flavor.id]
        verifylist = [('flavor', self.flavor.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_flavor.assert_called_with(parsed_args.flavor, get_extra_specs=True, ignore_missing=False)
        self.compute_sdk_client.flavor_add_tenant_access.assert_not_called()
        self.assertIsNone(result)

    def test_flavor_set_description_api_newer(self):
        arglist = ['--description', 'description', self.flavor.id]
        verifylist = [('description', 'description'), ('flavor', self.flavor.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(sdk_utils, 'supports_microversion', return_value=True):
            result = self.cmd.take_action(parsed_args)
            self.compute_sdk_client.update_flavor.assert_called_with(flavor=self.flavor.id, description='description')
            self.assertIsNone(result)

    def test_flavor_set_description_api_older(self):
        arglist = ['--description', 'description', self.flavor.id]
        verifylist = [('description', 'description'), ('flavor', self.flavor.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(sdk_utils, 'supports_microversion', return_value=False):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_flavor_set_description_using_name_api_newer(self):
        arglist = ['--description', 'description', self.flavor.name]
        verifylist = [('description', 'description'), ('flavor', self.flavor.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(sdk_utils, 'supports_microversion', return_value=True):
            result = self.cmd.take_action(parsed_args)
            self.compute_sdk_client.update_flavor.assert_called_with(flavor=self.flavor.id, description='description')
            self.assertIsNone(result)

    def test_flavor_set_description_using_name_api_older(self):
        arglist = ['--description', 'description', self.flavor.name]
        verifylist = [('description', 'description'), ('flavor', self.flavor.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(sdk_utils, 'supports_microversion', return_value=False):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)