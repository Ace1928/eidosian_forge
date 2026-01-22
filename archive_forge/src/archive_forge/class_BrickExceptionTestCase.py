from os_brick import exception
from os_brick.tests import base
class BrickExceptionTestCase(base.TestCase):

    def test_default_error_msg(self):

        class FakeBrickException(exception.BrickException):
            message = 'default message'
        exc = FakeBrickException()
        self.assertEqual(str(exc), 'default message')

    def test_error_msg(self):
        self.assertEqual(str(exception.BrickException('test')), 'test')

    def test_default_error_msg_with_kwargs(self):

        class FakeBrickException(exception.BrickException):
            message = 'default message: %(code)s'
        exc = FakeBrickException(code=500)
        self.assertEqual(str(exc), 'default message: 500')

    def test_error_msg_exception_with_kwargs(self):

        class FakeBrickException(exception.BrickException):
            message = 'default message: %(mispelled_code)s'
        exc = FakeBrickException(code=500)
        self.assertEqual(str(exc), 'default message: %(mispelled_code)s')

    def test_default_error_code(self):

        class FakeBrickException(exception.BrickException):
            code = 404
        exc = FakeBrickException()
        self.assertEqual(exc.kwargs['code'], 404)

    def test_error_code_from_kwarg(self):

        class FakeBrickException(exception.BrickException):
            code = 500
        exc = FakeBrickException(code=404)
        self.assertEqual(exc.kwargs['code'], 404)