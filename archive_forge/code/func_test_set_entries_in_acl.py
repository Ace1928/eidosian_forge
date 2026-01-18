from unittest import mock
import ddt
from os_win.tests.unit import test_base
from os_win.utils import _acl_utils
from os_win.utils.winapi import constants as w_const
@mock.patch.object(_acl_utils.ACLUtils, '_get_void_pp')
def test_set_entries_in_acl(self, mock_get_void_pp):
    new_acl = mock_get_void_pp.return_value
    returned_acl = self._acl_utils.set_entries_in_acl(mock.sentinel.entry_count, mock.sentinel.entry_list, mock.sentinel.old_acl)
    self.assertEqual(new_acl, returned_acl)
    self._mock_run.assert_called_once_with(_acl_utils.advapi32.SetEntriesInAclW, mock.sentinel.entry_count, mock.sentinel.entry_list, mock.sentinel.old_acl, new_acl)
    mock_get_void_pp.assert_called_once_with()