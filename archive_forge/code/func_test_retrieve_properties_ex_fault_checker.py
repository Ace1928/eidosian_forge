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
def test_retrieve_properties_ex_fault_checker(self):
    fault_list = ['FileFault', 'VimFault']
    missing_set = []
    for fault in fault_list:
        missing_elem = mock.Mock()
        missing_elem.fault.fault.__class__.__name__ = fault
        missing_set.append(missing_elem)
    obj_cont = mock.Mock()
    obj_cont.missingSet = missing_set
    response = mock.Mock()
    response.objects = [obj_cont]
    ex = self.assertRaises(exceptions.VimFaultException, service.Service._retrieve_properties_ex_fault_checker, response)
    self.assertEqual(fault_list, ex.fault_list)