import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_release_lease_incomplete_transfer(self):
    session = mock.Mock()
    handle = rw_handles.VmdkHandle(session, None, 'fake-url', None)
    handle._get_progress = mock.Mock(return_value=99)
    session.invoke_api = mock.Mock()
    handle._release_lease()
    session.invoke_api.assert_called_with(handle._session.vim, 'HttpNfcLeaseAbort', handle._lease)