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
def test_request_handler_with_web_fault(self):
    managed_object = 'VirtualMachine'
    fault_list = ['Fault']
    doc = mock.Mock()

    def side_effect(mo, **kwargs):
        self.assertEqual(managed_object, vim_util.get_moref_type(mo))
        self.assertEqual(managed_object, vim_util.get_moref_value(mo))
        fault = mock.Mock(faultstring='MyFault')
        fault_children = mock.Mock()
        fault_children.name = 'name'
        fault_children.getText.return_value = 'value'
        child = mock.Mock()
        child.get.return_value = fault_list[0]
        child.getChildren.return_value = [fault_children]
        detail = mock.Mock()
        detail.getChildren.return_value = [child]
        doc.childAtPath.return_value = detail
        raise suds.WebFault(fault, doc)
    svc_obj = service.Service()
    service_mock = svc_obj.client.service
    setattr(service_mock, 'powerOn', side_effect)
    ex = self.assertRaises(exceptions.VimFaultException, svc_obj.powerOn, managed_object)
    self.assertEqual(fault_list, ex.fault_list)
    self.assertEqual({'name': 'value'}, ex.details)
    self.assertEqual('MyFault', ex.msg)
    doc.childAtPath.assert_called_once_with('/detail')