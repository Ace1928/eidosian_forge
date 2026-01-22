import http.client
from oslo_serialization import jsonutils
import webtest
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
def assertValidErrorResponse(self, response, expected_status=http.client.BAD_REQUEST):
    """Verify that the error response is valid.

        Subclasses can override this function based on the expected response.

        """
    self.assertEqual(expected_status, response.status_code)
    error = response.result['error']
    self.assertEqual(response.status_code, error['code'])
    self.assertIsNotNone(error.get('title'))