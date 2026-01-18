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
def test_request_handler_with_attribute_error(self):
    managed_object = 'VirtualMachine'
    svc_obj = service.Service()
    service_mock = mock.Mock(spec=service.Service)
    svc_obj.client.service = service_mock
    self.assertRaises(exceptions.VimAttributeException, svc_obj.powerOn, managed_object)