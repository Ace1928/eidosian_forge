from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.ddt
class TestShareAccessDelete(TestShareAccess):

    def setUp(self):
        super(TestShareAccessDelete, self).setUp()
        self.share = manila_fakes.FakeShare.create_one_share(methods={'deny': None})
        self.shares_mock.get.return_value = self.share
        self.access_rule = manila_fakes.FakeShareAccessRule.create_one_access_rule(attrs={'share_id': self.share.id})
        self.cmd = osc_share_access_rules.ShareAccessDeny(self.app, None)

    @ddt.data(True, False)
    def test_share_access_delete(self, unrestrict):
        arglist = [self.share.id, self.access_rule.id]
        verifylist = [('share', self.share.id), ('id', self.access_rule.id)]
        deny_kwargs = {}
        if unrestrict:
            arglist.append('--unrestrict')
            verifylist.append(('unrestrict', unrestrict))
            deny_kwargs['unrestrict'] = unrestrict
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.shares_mock.get.assert_called_with(self.share.id)
        self.share.deny.assert_called_with(self.access_rule.id, **deny_kwargs)
        self.assertIsNone(result)

    def test_share_access_delete_unrestrict_not_available(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.79')
        arglist = [self.share.id, self.access_rule.id, '--unrestrict']
        verifylist = [('share', self.share.id), ('id', self.access_rule.id), ('unrestrict', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_access_delete_wait(self):
        arglist = [self.share.id, self.access_rule.id, '--wait']
        verifylist = [('share', self.share.id), ('id', self.access_rule.id), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
            result = self.cmd.take_action(parsed_args)
            self.shares_mock.get.assert_called_with(self.share.id)
            self.share.deny.assert_called_with(self.access_rule.id)
            self.assertIsNone(result)

    def test_share_access_delete_wait_error(self):
        arglist = [self.share.id, self.access_rule.id, '--wait']
        verifylist = [('share', self.share.id), ('id', self.access_rule.id), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=False):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)