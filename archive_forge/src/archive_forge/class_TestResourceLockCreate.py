from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import resource_locks as osc_resource_locks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestResourceLockCreate(TestResourceLock):

    def setUp(self):
        super(TestResourceLockCreate, self).setUp()
        self.share = manila_fakes.FakeShare.create_one_share()
        self.shares_mock.create.return_value = self.share
        self.shares_mock.get.return_value = self.share
        self.lock = manila_fakes.FakeResourceLock.create_one_lock(attrs={'resource_id': self.share.id})
        self.locks_mock.get.return_value = self.lock
        self.locks_mock.create.return_value = self.lock
        self.cmd = osc_resource_locks.CreateResourceLock(self.app, None)
        self.data = tuple(self.lock._info.values())
        self.columns = tuple(self.lock._info.keys())

    def test_share_lock_create_missing_required_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_lock_create(self):
        arglist = ['--resource-action', 'revert_to_snapshot', '--lock-reason', 'you cannot go back in time', self.share.id, 'share']
        verifylist = [('resource', self.share.id), ('resource_type', 'share'), ('resource_action', 'revert_to_snapshot'), ('lock_reason', 'you cannot go back in time')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.locks_mock.create.assert_called_with(self.share.id, 'share', 'revert_to_snapshot', 'you cannot go back in time')
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)