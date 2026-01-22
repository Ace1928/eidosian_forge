from unittest import mock
from openstackclient.common import extension
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class TestExtensionList(TestExtension):
    columns = ('Name', 'Alias', 'Description')
    long_columns = ('Name', 'Alias', 'Description', 'Namespace', 'Updated At', 'Links')
    volume_extension = volume_fakes.create_one_extension()
    identity_extension = identity_fakes.FakeExtension.create_one_extension()
    compute_extension = compute_fakes.create_one_extension()
    network_extension = network_fakes.create_one_extension()

    def setUp(self):
        super().setUp()
        self.identity_extensions_mock.list.return_value = [self.identity_extension]
        self.compute_extensions_mock.return_value = [self.compute_extension]
        self.volume_extensions_mock.return_value = [self.volume_extension]
        self.network_client.extensions.return_value = [self.network_extension]
        self.cmd = extension.ListExtension(self.app, None)

    def _test_extension_list_helper(self, arglist, verifylist, expected_data, long=False):
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        if long:
            self.assertEqual(self.long_columns, columns)
        else:
            self.assertEqual(self.columns, columns)
        self.assertEqual(expected_data, tuple(data))

    def test_extension_list_no_options(self):
        arglist = []
        verifylist = []
        datalist = ((self.identity_extension.name, self.identity_extension.alias, self.identity_extension.description), (self.compute_extension.name, self.compute_extension.alias, self.compute_extension.description), (self.volume_extension.name, self.volume_extension.alias, self.volume_extension.description), (self.network_extension.name, self.network_extension.alias, self.network_extension.description))
        self._test_extension_list_helper(arglist, verifylist, datalist)
        self.identity_extensions_mock.list.assert_called_with()
        self.compute_extensions_mock.assert_called_with()
        self.volume_extensions_mock.assert_called_with()
        self.network_client.extensions.assert_called_with()

    def test_extension_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        datalist = ((self.identity_extension.name, self.identity_extension.alias, self.identity_extension.description, self.identity_extension.namespace, '', self.identity_extension.links), (self.compute_extension.name, self.compute_extension.alias, self.compute_extension.description, self.compute_extension.namespace, self.compute_extension.updated_at, self.compute_extension.links), (self.volume_extension.name, self.volume_extension.alias, self.volume_extension.description, '', self.volume_extension.updated_at, self.volume_extension.links), (self.network_extension.name, self.network_extension.alias, self.network_extension.description, '', self.network_extension.updated_at, self.network_extension.links))
        self._test_extension_list_helper(arglist, verifylist, datalist, True)
        self.identity_extensions_mock.list.assert_called_with()
        self.compute_extensions_mock.assert_called_with()
        self.volume_extensions_mock.assert_called_with()
        self.network_client.extensions.assert_called_with()

    def test_extension_list_identity(self):
        arglist = ['--identity']
        verifylist = [('identity', True)]
        datalist = ((self.identity_extension.name, self.identity_extension.alias, self.identity_extension.description),)
        self._test_extension_list_helper(arglist, verifylist, datalist)
        self.identity_extensions_mock.list.assert_called_with()

    def test_extension_list_network(self):
        arglist = ['--network']
        verifylist = [('network', True)]
        datalist = ((self.network_extension.name, self.network_extension.alias, self.network_extension.description),)
        self._test_extension_list_helper(arglist, verifylist, datalist)
        self.network_client.extensions.assert_called_with()

    def test_extension_list_network_with_long(self):
        arglist = ['--network', '--long']
        verifylist = [('network', True), ('long', True)]
        datalist = ((self.network_extension.name, self.network_extension.alias, self.network_extension.description, '', self.network_extension.updated_at, self.network_extension.links),)
        self._test_extension_list_helper(arglist, verifylist, datalist, long=True)
        self.network_client.extensions.assert_called_with()

    def test_extension_list_compute(self):
        arglist = ['--compute']
        verifylist = [('compute', True)]
        datalist = ((self.compute_extension.name, self.compute_extension.alias, self.compute_extension.description),)
        self._test_extension_list_helper(arglist, verifylist, datalist)
        self.compute_extensions_mock.assert_called_with()

    def test_extension_list_compute_and_network(self):
        arglist = ['--compute', '--network']
        verifylist = [('compute', True), ('network', True)]
        datalist = ((self.compute_extension.name, self.compute_extension.alias, self.compute_extension.description), (self.network_extension.name, self.network_extension.alias, self.network_extension.description))
        self._test_extension_list_helper(arglist, verifylist, datalist)
        self.compute_extensions_mock.assert_called_with()
        self.network_client.extensions.assert_called_with()

    def test_extension_list_volume(self):
        arglist = ['--volume']
        verifylist = [('volume', True)]
        datalist = ((self.volume_extension.name, self.volume_extension.alias, self.volume_extension.description),)
        self._test_extension_list_helper(arglist, verifylist, datalist)
        self.volume_extensions_mock.assert_called_with()