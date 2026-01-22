import unittest
from numba.tests.support import TestCase, skip_unless_typeguard
@skip_unless_typeguard
class TestTypeGuard(TestCase):

    def setUp(self):
        super().setUp()
        import typeguard
        self._exception_type = getattr(typeguard, 'TypeCheckError', TypeError)

    def test_check_args(self):
        with self.assertRaises(self._exception_type):
            guard_args(float(1.2))

    def test_check_ret(self):
        with self.assertRaises(self._exception_type):
            guard_ret(float(1.2))

    def test_check_does_not_work_with_inner_func(self):

        def guard(val: int) -> int:
            return
        guard(float(1.2))