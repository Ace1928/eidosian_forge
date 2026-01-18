from keystoneauth1 import exceptions
from keystoneauth1.tests.unit import utils
def test_using_default_message(self):
    exc = exceptions.AuthorizationFailure()
    self.assertEqual(exceptions.AuthorizationFailure.message, exc.message)