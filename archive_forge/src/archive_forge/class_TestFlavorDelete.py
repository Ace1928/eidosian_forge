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
class TestFlavorDelete(TestFlavor):
    flavors = compute_fakes.create_flavors(count=2)

    def setUp(self):
        super(TestFlavorDelete, self).setUp()
        self.compute_sdk_client.delete_flavor.return_value = None
        self.cmd = flavor.DeleteFlavor(self.app, None)

    def test_flavor_delete(self):
        arglist = [self.flavors[0].id]
        verifylist = [('flavor', [self.flavors[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.compute_sdk_client.find_flavor.return_value = self.flavors[0]
        result = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.find_flavor.assert_called_with(self.flavors[0].id, ignore_missing=False)
        self.compute_sdk_client.delete_flavor.assert_called_with(self.flavors[0].id)
        self.assertIsNone(result)

    def test_delete_multiple_flavors(self):
        arglist = []
        for f in self.flavors:
            arglist.append(f.id)
        verifylist = [('flavor', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.compute_sdk_client.find_flavor.side_effect = self.flavors
        result = self.cmd.take_action(parsed_args)
        find_calls = [mock.call(i.id, ignore_missing=False) for i in self.flavors]
        delete_calls = [mock.call(i.id) for i in self.flavors]
        self.compute_sdk_client.find_flavor.assert_has_calls(find_calls)
        self.compute_sdk_client.delete_flavor.assert_has_calls(delete_calls)
        self.assertIsNone(result)

    def test_multi_flavors_delete_with_exception(self):
        arglist = [self.flavors[0].id, 'unexist_flavor']
        verifylist = [('flavor', [self.flavors[0].id, 'unexist_flavor'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.compute_sdk_client.find_flavor.side_effect = [self.flavors[0], sdk_exceptions.ResourceNotFound]
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 flavors failed to delete.', str(e))
        find_calls = [mock.call(self.flavors[0].id, ignore_missing=False), mock.call('unexist_flavor', ignore_missing=False)]
        delete_calls = [mock.call(self.flavors[0].id)]
        self.compute_sdk_client.find_flavor.assert_has_calls(find_calls)
        self.compute_sdk_client.delete_flavor.assert_has_calls(delete_calls)