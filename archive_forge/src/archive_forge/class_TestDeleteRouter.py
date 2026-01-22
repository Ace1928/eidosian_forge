from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteRouter(TestRouter):
    _routers = network_fakes.FakeRouter.create_routers(count=2)

    def setUp(self):
        super(TestDeleteRouter, self).setUp()
        self.network_client.delete_router = mock.Mock(return_value=None)
        self.network_client.find_router = network_fakes.FakeRouter.get_routers(self._routers)
        self.cmd = router.DeleteRouter(self.app, self.namespace)

    def test_router_delete(self):
        arglist = [self._routers[0].name]
        verifylist = [('router', [self._routers[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_router.assert_called_once_with(self._routers[0])
        self.assertIsNone(result)

    def test_multi_routers_delete(self):
        arglist = []
        verifylist = []
        for r in self._routers:
            arglist.append(r.name)
        verifylist = [('router', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for r in self._routers:
            calls.append(call(r))
        self.network_client.delete_router.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_multi_routers_delete_with_exception(self):
        arglist = [self._routers[0].name, 'unexist_router']
        verifylist = [('router', [self._routers[0].name, 'unexist_router'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self._routers[0], exceptions.CommandError]
        self.network_client.find_router = mock.Mock(side_effect=find_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 routers failed to delete.', str(e))
        self.network_client.find_router.assert_any_call(self._routers[0].name, ignore_missing=False)
        self.network_client.find_router.assert_any_call('unexist_router', ignore_missing=False)
        self.network_client.delete_router.assert_called_once_with(self._routers[0])