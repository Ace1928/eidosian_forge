from keystoneauth1 import exceptions
from keystoneauth1.tests.unit import utils
class ExceptionTests(utils.TestCase):

    def test_clientexception_with_message(self):
        test_message = 'Unittest exception message.'
        exc = exceptions.ClientException(message=test_message)
        self.assertEqual(test_message, exc.message)

    def test_clientexception_with_no_message(self):
        exc = exceptions.ClientException()
        self.assertEqual(exceptions.ClientException.__name__, exc.message)

    def test_using_default_message(self):
        exc = exceptions.AuthorizationFailure()
        self.assertEqual(exceptions.AuthorizationFailure.message, exc.message)