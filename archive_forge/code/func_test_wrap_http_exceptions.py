from unittest import mock
import keystoneauth1.exceptions.http as ks_exceptions
import osc_lib.exceptions as exceptions
import oslotest.base as base
import requests
import simplejson as json
from osc_placement import http
from osc_placement import version
from oslo_serialization import jsonutils
def test_wrap_http_exceptions(self):

    def go():
        with http._wrap_http_exceptions():
            error = {'errors': [{'status': 404, 'detail': 'The resource could not be found.\n\nNo resource provider with uuid 123 found for delete'}]}
            response = mock.Mock(content=json.dumps(error))
            raise ks_exceptions.NotFound(response=response)
    exc = self.assertRaises(exceptions.NotFound, go)
    self.assertEqual(404, exc.http_status)
    self.assertIn('No resource provider with uuid 123 found', str(exc))