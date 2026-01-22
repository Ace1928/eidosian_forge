from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils
class UnauthenticatedTestCase(utils.TestCase):
    """Class used as base for unauthenticated calls."""
    TEST_ROOT_URL = 'http://127.0.0.1:5000/'
    TEST_URL = '%s%s' % (TEST_ROOT_URL, 'v2.0')
    TEST_ROOT_ADMIN_URL = 'http://127.0.0.1:35357/'
    TEST_ADMIN_URL = '%s%s' % (TEST_ROOT_ADMIN_URL, 'v2.0')