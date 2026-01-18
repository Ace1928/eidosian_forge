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
def test_retrieve_properties_ex_fault_checker_with_empty_response(self):
    ex = self.assertRaises(exceptions.VimFaultException, service.Service._retrieve_properties_ex_fault_checker, None)
    self.assertEqual([exceptions.NOT_AUTHENTICATED], ex.fault_list)