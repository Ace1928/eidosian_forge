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
def test_request_handler_with_address_in_use_error(self):
    self._test_request_handler_with_exception(service.ADDRESS_IN_USE_ERROR, exceptions.VimSessionOverLoadException)