import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_update_progress_with_error(self):
    session = mock.Mock()
    handle = rw_handles.VmdkHandle(session, None, 'fake-url', None)
    handle._get_progress = mock.Mock(return_value=0)
    session.invoke_api.side_effect = exceptions.VimException(None)
    self.assertRaises(exceptions.VimException, handle.update_progress)