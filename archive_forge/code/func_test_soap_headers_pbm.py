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
def test_soap_headers_pbm(self):

    def fake_set_options(*args, **kwargs):
        headers = kwargs['soapheaders']
        self.assertEqual(2, len(headers))
        self.assertEqual('vc-session-cookie', headers[0].getText())
        self.assertEqual('fira-12345', headers[1].getText())
    svc_obj = service.Service()
    svc_obj._vc_session_cookie = 'vc-session-cookie'
    setattr(svc_obj.client, 'set_options', fake_set_options)
    svc_obj._set_soap_headers('fira-12345')