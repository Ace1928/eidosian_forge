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
def test_request_handler_with_empty_web_fault_doc(self):

    def side_effect(mo, **kwargs):
        fault = mock.Mock(faultstring='MyFault')
        raise suds.WebFault(fault, None)
    svc_obj = service.Service()
    service_mock = svc_obj.client.service
    setattr(service_mock, 'powerOn', side_effect)
    ex = self.assertRaises(exceptions.VimFaultException, svc_obj.powerOn, 'VirtualMachine')
    self.assertEqual([], ex.fault_list)
    self.assertEqual({}, ex.details)
    self.assertEqual('MyFault', ex.msg)