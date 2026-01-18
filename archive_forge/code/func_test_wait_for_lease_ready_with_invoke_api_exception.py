from datetime import datetime
from unittest import mock
from eventlet import greenthread
from oslo_context import context
import suds
from oslo_vmware import api
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_wait_for_lease_ready_with_invoke_api_exception(self):
    api_session = self._create_api_session(True)
    api_session.invoke_api = mock.Mock(side_effect=exceptions.VimException(None))
    lease = mock.Mock()
    self.assertRaises(exceptions.VimException, api_session.wait_for_lease_ready, lease)
    api_session.invoke_api.assert_called_once_with(vim_util, 'get_object_property', api_session.vim, lease, 'state', skip_op_id=True)