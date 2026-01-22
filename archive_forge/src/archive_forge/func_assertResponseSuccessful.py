import http.client
from oslo_serialization import jsonutils
import webtest
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
def assertResponseSuccessful(self, response):
    """Assert that a status code lies inside the 2xx range.

        :param response: :py:class:`httplib.HTTPResponse` to be
          verified to have a status code between 200 and 299.

        example::

             self.assertResponseSuccessful(response)
        """
    self.assertTrue(200 <= response.status_code <= 299, 'Status code %d is outside of the expected range (2xx)\n\n%s' % (response.status, response.body))