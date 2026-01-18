import copy
from unittest import mock
from oslo_i18n import fixture as i18n_fixture
import suds
from oslo_vmware import exceptions
from oslo_vmware.tests import base
from oslo_vmware import vim
@mock.patch.object(vim.Vim, '__getattr__', autospec=True)
def test_service_content(self, getattr_mock):
    getattr_ret = mock.Mock()
    getattr_mock.side_effect = lambda *args: getattr_ret
    vim_obj = vim.Vim()
    vim_obj.service_content
    getattr_mock.assert_called_once_with(vim_obj, 'RetrieveServiceContent')
    getattr_ret.assert_called_once_with('ServiceInstance')
    self.assertEqual(self.SudsClientMock.return_value, vim_obj.client)
    self.assertEqual(getattr_ret.return_value, vim_obj.service_content)