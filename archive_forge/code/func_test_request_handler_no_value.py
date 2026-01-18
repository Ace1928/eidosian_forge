import http.client as httplib
import io
from unittest import mock
import ddt
import requests
import suds
from oslo_vmware import exceptions
from oslo_vmware import service
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch.object(vim_util, 'get_moref', return_value=None)
def test_request_handler_no_value(self, mock_moref):
    managed_object = 'VirtualMachine'
    svc_obj = service.Service()
    ret = svc_obj.UnregisterVM(managed_object)
    self.assertIsNone(ret)