import oslo_messaging
from oslo_messaging.tests import utils as test_utils
class TestExpectedExceptions(test_utils.BaseTestCase):

    def test_exception(self):
        e = None
        try:
            try:
                raise ValueError()
            except Exception:
                raise oslo_messaging.ExpectedException()
        except oslo_messaging.ExpectedException as e:
            self.assertIsInstance(e, oslo_messaging.ExpectedException)
            self.assertTrue(hasattr(e, 'exc_info'))
            self.assertIsInstance(e.exc_info[1], ValueError)

    def test_decorator_expected(self):

        class FooException(Exception):
            pass

        @oslo_messaging.expected_exceptions(FooException)
        def naughty():
            raise FooException()
        self.assertRaises(oslo_messaging.ExpectedException, naughty)

    def test_decorator_expected_subclass(self):

        class FooException(Exception):
            pass

        class BarException(FooException):
            pass

        @oslo_messaging.expected_exceptions(FooException)
        def naughty():
            raise BarException()
        self.assertRaises(oslo_messaging.ExpectedException, naughty)

    def test_decorator_unexpected(self):

        class FooException(Exception):
            pass

        @oslo_messaging.expected_exceptions(FooException)
        def really_naughty():
            raise ValueError()
        self.assertRaises(ValueError, really_naughty)