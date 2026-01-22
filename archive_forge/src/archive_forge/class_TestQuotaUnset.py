import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import quota
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestQuotaUnset(TestQuota):
    PARAMETERS = ('loadbalancer', 'listener', 'pool', 'member', 'healthmonitor', 'l7policy', 'l7rule')

    def setUp(self):
        super().setUp()
        self.cmd = quota.UnsetQuota(self.app, None)

    def test_quota_unset_loadbalancer(self):
        self._test_quota_unset_param('loadbalancer')

    def test_quota_unset_listener(self):
        self._test_quota_unset_param('listener')

    def test_quota_unset_pool(self):
        self._test_quota_unset_param('pool')

    def test_quota_unset_health_monitor(self):
        self._test_quota_unset_param('healthmonitor')

    def test_quota_unset_member(self):
        self._test_quota_unset_param('member')

    def test_quota_unset_l7policy(self):
        self._test_quota_unset_param('l7policy')

    def test_quota_unset_l7rule(self):
        self._test_quota_unset_param('l7rule')

    @mock.patch('octaviaclient.osc.v2.utils.get_resource_id')
    def _test_quota_unset_param(self, param, mock_get_resource):
        self.api_mock.quota_set.reset_mock()
        mock_get_resource.return_value = self._qt.project_id
        arg_param = param.replace('_', '-') if '_' in param else param
        arglist = [self._qt.project_id, '--%s' % arg_param]
        ref_body = {'quota': {param: None}}
        verifylist = [('project', self._qt.project_id)]
        for ref_param in self.PARAMETERS:
            verifylist.append((ref_param, param == ref_param))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.quota_set.assert_called_once_with(self._qt.project_id, json=ref_body)

    @mock.patch('octaviaclient.osc.v2.utils.get_resource_id')
    def test_quota_unset_all(self, mock_get_resource):
        self.api_mock.quota_set.reset_mock()
        mock_get_resource.return_value = self._qt.project_id
        ref_body = {'quota': {x: None for x in self.PARAMETERS}}
        arglist = [self._qt.project_id]
        for ref_param in self.PARAMETERS:
            arg_param = ref_param.replace('_', '-') if '_' in ref_param else ref_param
            arglist.append('--%s' % arg_param)
        verifylist = list(zip(self.PARAMETERS, [True] * len(self.PARAMETERS)))
        verifylist = [('project', self._qt.project_id)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.quota_set.assert_called_once_with(self._qt.project_id, json=ref_body)

    def test_quota_unset_none(self):
        self.api_mock.quota_set.reset_mock()
        arglist = [self._qt.project_id]
        verifylist = list(zip(self.PARAMETERS, [False] * len(self.PARAMETERS)))
        verifylist = [('project', self._qt.project_id)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.quota_set.assert_not_called()