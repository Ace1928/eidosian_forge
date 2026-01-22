import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import accounts
class AccountTest(testtools.TestCase):

    def setUp(self):
        super(AccountTest, self).setUp()
        self.orig__init = accounts.Account.__init__
        accounts.Account.__init__ = mock.Mock(return_value=None)
        self.account = accounts.Account()

    def tearDown(self):
        super(AccountTest, self).tearDown()
        accounts.Account.__init__ = self.orig__init

    def test___repr__(self):
        self.account.name = 'account-1'
        self.assertEqual('<Account: account-1>', self.account.__repr__())