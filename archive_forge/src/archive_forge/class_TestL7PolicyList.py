import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7policy
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestL7PolicyList(TestL7Policy):

    def setUp(self):
        super().setUp()
        self.datalist = (tuple((attr_consts.L7POLICY_ATTRS[k] for k in self.columns)),)
        self.cmd = l7policy.ListL7Policy(self.app, None)

    def test_l7policy_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_list.assert_called_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    @mock.patch('octaviaclient.osc.v2.utils.get_l7policy_attrs')
    def test_l7policy_list_by_listener(self, mock_l7policy_attrs):
        mock_l7policy_attrs.return_value = {'listener_id': self._l7po.listener_id}
        arglist = ['--listener', 'mock_li_id']
        verifylist = [('listener', 'mock_li_id')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_list.assert_called_with(listener_id=self._l7po.listener_id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_l7policy_list_with_tags(self):
        arglist = ['--tags', 'foo,bar']
        verifylist = [('tags', ['foo', 'bar'])]
        expected_attrs = {'tags': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_l7policy_list_with_any_tags(self):
        arglist = ['--any-tags', 'foo,bar']
        verifylist = [('any_tags', ['foo', 'bar'])]
        expected_attrs = {'tags-any': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_l7policy_list_with_not_tags(self):
        arglist = ['--not-tags', 'foo,bar']
        verifylist = [('not_tags', ['foo', 'bar'])]
        expected_attrs = {'not-tags': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_l7policy_list_with_not_any_tags(self):
        arglist = ['--not-any-tags', 'foo,bar']
        verifylist = [('not_any_tags', ['foo', 'bar'])]
        expected_attrs = {'not-tags-any': ['foo', 'bar']}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.api_mock.l7policy_list.assert_called_with(**expected_attrs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))