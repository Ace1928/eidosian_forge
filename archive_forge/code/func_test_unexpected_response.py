from unittest import mock
import keystoneauth1.exceptions.http as ks_exceptions
import osc_lib.exceptions as exceptions
import oslotest.base as base
import requests
import simplejson as json
from osc_placement import http
from osc_placement import version
from oslo_serialization import jsonutils
def test_unexpected_response(self):

    def go():
        with http._wrap_http_exceptions():
            raise ks_exceptions.InternalServerError()
    exc = self.assertRaises(ks_exceptions.InternalServerError, go)
    self.assertEqual(500, exc.http_status)
    self.assertIn('Internal Server Error (HTTP 500)', str(exc))