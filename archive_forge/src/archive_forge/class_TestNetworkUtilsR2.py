from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
class TestNetworkUtilsR2(test_base.OsWinBaseTestCase):

    def setUp(self):
        super(TestNetworkUtilsR2, self).setUp()
        self.netutils = networkutils.NetworkUtilsR2()
        self.netutils._conn_attr = mock.MagicMock()

    @mock.patch.object(networkutils.NetworkUtilsR2, '_create_default_setting_data')
    def test_create_security_acl(self, mock_create_default_setting_data):
        sg_rule = mock.MagicMock()
        sg_rule.to_dict.return_value = {}
        acl = self.netutils._create_security_acl(sg_rule, mock.sentinel.weight)
        self.assertEqual(mock.sentinel.weight, acl.Weight)

    def test_get_new_weights_no_acls_deny(self):
        mock_rule = mock.MagicMock(Action=self.netutils._ACL_ACTION_DENY)
        actual = self.netutils._get_new_weights([mock_rule], [])
        self.assertEqual([1], actual)

    def test_get_new_weights_no_acls_allow(self):
        mock_rule = mock.MagicMock(Action=self.netutils._ACL_ACTION_ALLOW)
        actual = self.netutils._get_new_weights([mock_rule, mock_rule], [])
        expected = [self.netutils._MAX_WEIGHT - 1, self.netutils._MAX_WEIGHT - 2]
        self.assertEqual(expected, actual)

    def test_get_new_weights_deny(self):
        mock_rule = mock.MagicMock(Action=self.netutils._ACL_ACTION_DENY)
        mockacl1 = mock.MagicMock(Action=self.netutils._ACL_ACTION_DENY, Weight=1)
        mockacl2 = mock.MagicMock(Action=self.netutils._ACL_ACTION_DENY, Weight=3)
        actual = self.netutils._get_new_weights([mock_rule, mock_rule], [mockacl1, mockacl2])
        self.assertEqual([2, 4], actual)

    def test_get_new_weights_allow(self):
        mock_rule = mock.MagicMock(Action=self.netutils._ACL_ACTION_ALLOW)
        mockacl = mock.MagicMock(Action=self.netutils._ACL_ACTION_ALLOW, Weight=self.netutils._MAX_WEIGHT - 3)
        actual = self.netutils._get_new_weights([mock_rule, mock_rule], [mockacl])
        expected = [self.netutils._MAX_WEIGHT - 4, self.netutils._MAX_WEIGHT - 5]
        self.assertEqual(expected, actual)

    def test_get_new_weights_search_available(self):
        mock_rule = mock.MagicMock(Action=self.netutils._ACL_ACTION_ALLOW)
        mockacl1 = mock.MagicMock(Action=self.netutils._ACL_ACTION_ALLOW, Weight=self.netutils._REJECT_ACLS_COUNT + 1)
        mockacl2 = mock.MagicMock(Action=self.netutils._ACL_ACTION_ALLOW, Weight=self.netutils._MAX_WEIGHT - 1)
        actual = self.netutils._get_new_weights([mock_rule], [mockacl1, mockacl2])
        self.assertEqual([self.netutils._MAX_WEIGHT - 2], actual)