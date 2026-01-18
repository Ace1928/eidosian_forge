import http.client
import httplib2
from oslo_utils.fixture import uuidsentinel as uuids
from glance.tests import functional
Provide a basic smoke test to ensure CORS middleware is active.

    The tests below provide minimal confirmation that the CORS middleware
    is active, and may be configured. For comprehensive tests, please consult
    the test suite in oslo_middleware.
    