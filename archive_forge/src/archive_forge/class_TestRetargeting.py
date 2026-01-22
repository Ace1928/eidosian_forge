import unittest
from contextlib import contextmanager
from functools import cached_property
from numba import njit
from numba.core import errors, cpu, typing
from numba.core.descriptors import TargetDescriptor
from numba.core.dispatcher import TargetConfigurationStack
from numba.core.retarget import BasicRetarget
from numba.core.extending import overload
from numba.core.target_extension import (
class TestRetargeting(unittest.TestCase):

    def setUp(self):

        @njit(_target='cpu')
        def fixed_target(x):
            """
            This has a fixed target to "cpu".
            Cannot be used in CUSTOM_TARGET target.
            """
            return x + 10

        @njit
        def flex_call_fixed(x):
            """
            This has a flexible target, but uses a fixed target function.
            Cannot be used in CUSTOM_TARGET target.
            """
            return fixed_target(x) + 100

        @njit
        def flex_target(x):
            """
            This has a flexible target.
            Can be used in CUSTOM_TARGET target.
            """
            return x + 1000
        self.functions = locals()
        self.retarget = CustomCPURetarget()

    def switch_target(self):
        return TargetConfigurationStack.switch_target(self.retarget)

    @contextmanager
    def check_retarget_error(self):
        with self.assertRaises(errors.NumbaError) as raises:
            yield
        self.assertIn(f'{CUSTOM_TARGET} != cpu', str(raises.exception))

    def check_non_empty_cache(self):
        stats = self.retarget.cache.stats()
        self.assertGreater(stats['hit'] + stats['miss'], 0)

    def test_case0(self):
        fixed_target = self.functions['fixed_target']
        flex_target = self.functions['flex_target']

        @njit
        def foo(x):
            x = fixed_target(x)
            x = flex_target(x)
            return x
        r = foo(123)
        self.assertEqual(r, 123 + 10 + 1000)
        stats = self.retarget.cache.stats()
        self.assertEqual(stats, dict(hit=0, miss=0))

    def test_case1(self):
        flex_target = self.functions['flex_target']

        @njit
        def foo(x):
            x = flex_target(x)
            return x
        with self.switch_target():
            r = foo(123)
        self.assertEqual(r, 123 + 1000)
        self.check_non_empty_cache()

    def test_case2(self):
        """
        The non-nested call into fixed_target should raise error.
        """
        fixed_target = self.functions['fixed_target']
        flex_target = self.functions['flex_target']

        @njit
        def foo(x):
            x = fixed_target(x)
            x = flex_target(x)
            return x
        with self.check_retarget_error():
            with self.switch_target():
                foo(123)

    def test_case3(self):
        """
        The nested call into fixed_target should raise error
        """
        flex_target = self.functions['flex_target']
        flex_call_fixed = self.functions['flex_call_fixed']

        @njit
        def foo(x):
            x = flex_call_fixed(x)
            x = flex_target(x)
            return x
        with self.check_retarget_error():
            with self.switch_target():
                foo(123)

    def test_case4(self):
        """
        Same as case2 but flex_call_fixed() is invoked outside of CUSTOM_TARGET
        target before the switch_target.
        """
        flex_target = self.functions['flex_target']
        flex_call_fixed = self.functions['flex_call_fixed']
        r = flex_call_fixed(123)
        self.assertEqual(r, 123 + 100 + 10)

        @njit
        def foo(x):
            x = flex_call_fixed(x)
            x = flex_target(x)
            return x
        with self.check_retarget_error():
            with self.switch_target():
                foo(123)

    def test_case5(self):
        """
        Tests overload resolution with target switching
        """

        def overloaded_func(x):
            pass

        @overload(overloaded_func, target=CUSTOM_TARGET)
        def ol_overloaded_func_custom_target(x):

            def impl(x):
                return 62830
            return impl

        @overload(overloaded_func, target='cpu')
        def ol_overloaded_func_cpu(x):

            def impl(x):
                return 31415
            return impl

        @njit
        def flex_resolve_overload(x):
            return

        @njit
        def foo(x):
            return x + overloaded_func(x)
        r = foo(123)
        self.assertEqual(r, 123 + 31415)
        with self.switch_target():
            r = foo(123)
            self.assertEqual(r, 123 + 62830)
        self.check_non_empty_cache()