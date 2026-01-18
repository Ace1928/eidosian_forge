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
def test_request_handler_with_connection_error(self):
    managed_object = 'VirtualMachine'

    def side_effect(mo, **kwargs):
        self.assertEqual(managed_object, vim_util.get_moref_type(mo))
        self.assertEqual(managed_object, vim_util.get_moref_value(mo))
        raise requests.ConnectionError()
    svc_obj = service.Service()
    attr_name = 'powerOn'
    service_mock = svc_obj.client.service
    setattr(service_mock, attr_name, side_effect)
    self.assertRaises(exceptions.VimConnectionException, svc_obj.powerOn, managed_object)