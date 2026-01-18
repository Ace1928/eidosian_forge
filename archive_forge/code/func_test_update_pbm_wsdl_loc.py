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
def test_update_pbm_wsdl_loc(self):
    session = mock.Mock()
    session.key = '12345'
    api_session = self._create_api_session(False)
    self.assertIsNone(api_session._pbm_wsdl_loc)
    api_session.pbm_wsdl_loc_set('fake_wsdl')
    self.assertEqual('fake_wsdl', api_session._pbm_wsdl_loc)