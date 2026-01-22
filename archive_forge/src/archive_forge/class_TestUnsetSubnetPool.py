from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet_pool
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
class TestUnsetSubnetPool(TestSubnetPool):

    def setUp(self):
        super(TestUnsetSubnetPool, self).setUp()
        self._subnetpool = network_fakes.FakeSubnetPool.create_one_subnet_pool({'tags': ['green', 'red']})
        self.network_client.find_subnet_pool = mock.Mock(return_value=self._subnetpool)
        self.network_client.update_subnet_pool = mock.Mock(return_value=None)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.cmd = subnet_pool.UnsetSubnetPool(self.app, self.namespace)

    def _test_unset_tags(self, with_tags=True):
        if with_tags:
            arglist = ['--tag', 'red', '--tag', 'blue']
            verifylist = [('tags', ['red', 'blue'])]
            expected_args = ['green']
        else:
            arglist = ['--all-tag']
            verifylist = [('all_tag', True)]
            expected_args = []
        arglist.append(self._subnetpool.name)
        verifylist.append(('subnet_pool', self._subnetpool.name))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_subnet_pool.called)
        self.network_client.set_tags.assert_called_once_with(self._subnetpool, test_utils.CompareBySet(expected_args))
        self.assertIsNone(result)

    def test_unset_with_tags(self):
        self._test_unset_tags(with_tags=True)

    def test_unset_with_all_tag(self):
        self._test_unset_tags(with_tags=False)