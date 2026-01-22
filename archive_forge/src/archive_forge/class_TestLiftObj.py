import copy
import warnings
import numpy as np
import numba
from numba.core.transforms import find_setupwiths, with_lifting
from numba.core.withcontexts import bypass_context, call_context, objmode_context
from numba.core.bytecode import FunctionIdentity, ByteCode
from numba.core.interpreter import Interpreter
from numba.core import errors
from numba.core.registry import cpu_target
from numba.core.compiler import compile_ir, DEFAULT_FLAGS
from numba import njit, typeof, objmode, types
from numba.core.extending import overload
from numba.tests.support import (MemoryLeak, TestCase, captured_stdout,
from numba.core.utils import PYVERSION
from numba.experimental import jitclass
import unittest
class TestLiftObj(MemoryLeak, TestCase):

    def setUp(self):
        warnings.simplefilter('error', errors.NumbaWarning)

    def tearDown(self):
        warnings.resetwarnings()

    def assert_equal_return_and_stdout(self, pyfunc, *args):
        py_args = copy.deepcopy(args)
        c_args = copy.deepcopy(args)
        cfunc = njit(pyfunc)
        with captured_stdout() as stream:
            expect_res = pyfunc(*py_args)
            expect_out = stream.getvalue()
        cfunc.compile(tuple(map(typeof, c_args)))
        with captured_stdout() as stream:
            got_res = cfunc(*c_args)
            got_out = stream.getvalue()
        self.assertEqual(expect_out, got_out)
        self.assertPreciseEqual(expect_res, got_res)

    def test_lift_objmode_basic(self):

        def bar(ival):
            print('ival =', {'ival': ival // 2})

        def foo(ival):
            ival += 1
            with objmode_context:
                bar(ival)
            return ival + 1

        def foo_nonglobal(ival):
            ival += 1
            with numba.objmode:
                bar(ival)
            return ival + 1
        self.assert_equal_return_and_stdout(foo, 123)
        self.assert_equal_return_and_stdout(foo_nonglobal, 123)

    def test_lift_objmode_array_in(self):

        def bar(arr):
            print({'arr': arr // 2})
            arr *= 2

        def foo(nelem):
            arr = np.arange(nelem).astype(np.int64)
            with objmode_context:
                bar(arr)
            return arr + 1
        nelem = 10
        self.assert_equal_return_and_stdout(foo, nelem)

    def test_lift_objmode_define_new_unused(self):

        def bar(y):
            print(y)

        def foo(x):
            with objmode_context():
                y = 2 + x
                a = np.arange(y)
                bar(a)
            return x
        arg = 123
        self.assert_equal_return_and_stdout(foo, arg)

    def test_lift_objmode_return_simple(self):

        def inverse(x):
            print(x)
            return 1 / x

        def foo(x):
            with objmode_context(y='float64'):
                y = inverse(x)
            return (x, y)

        def foo_nonglobal(x):
            with numba.objmode(y='float64'):
                y = inverse(x)
            return (x, y)
        arg = 123
        self.assert_equal_return_and_stdout(foo, arg)
        self.assert_equal_return_and_stdout(foo_nonglobal, arg)

    def test_lift_objmode_return_array(self):

        def inverse(x):
            print(x)
            return 1 / x

        def foo(x):
            with objmode_context(y='float64[:]', z='int64'):
                y = inverse(x)
                z = int(y[0])
            return (x, y, z)
        arg = np.arange(1, 10, dtype=np.float64)
        self.assert_equal_return_and_stdout(foo, arg)

    @expected_failure_for_list_arg
    def test_lift_objmode_using_list(self):

        def foo(x):
            with objmode_context(y='float64[:]'):
                print(x)
                x[0] = 4
                print(x)
                y = [1, 2, 3] + x
                y = np.asarray([1 / i for i in y])
            return (x, y)
        arg = [1, 2, 3]
        self.assert_equal_return_and_stdout(foo, arg)

    def test_lift_objmode_var_redef(self):

        def foo(x):
            for x in range(x):
                pass
            if x:
                x += 1
            with objmode_context(x='intp'):
                print(x)
                x -= 1
                print(x)
                for i in range(x):
                    x += i
                    print(x)
            return x
        arg = 123
        self.assert_equal_return_and_stdout(foo, arg)

    @expected_failure_for_list_arg
    def test_case01_mutate_list_ahead_of_ctx(self):

        def foo(x, z):
            x[2] = z
            with objmode_context():
                print(x)
            with objmode_context():
                x[2] = 2 * z
                print(x)
            return x
        self.assert_equal_return_and_stdout(foo, [1, 2, 3], 15)

    def test_case02_mutate_array_ahead_of_ctx(self):

        def foo(x, z):
            x[2] = z
            with objmode_context():
                print(x)
            with objmode_context():
                x[2] = 2 * z
                print(x)
            return x
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x, 15)

    @expected_failure_for_list_arg
    def test_case03_create_and_mutate(self):

        def foo(x):
            with objmode_context(y='List(int64)'):
                y = [1, 2, 3]
            with objmode_context():
                y[2] = 10
            return y
        self.assert_equal_return_and_stdout(foo, 1)

    def test_case04_bogus_variable_type_info(self):

        def foo(x):
            with objmode_context(k='float64[:]'):
                print(x)
            return x
        x = np.array([1, 2, 3])
        cfoo = njit(foo)
        with self.assertRaises(errors.TypingError) as raises:
            cfoo(x)
        self.assertIn('Invalid type annotation on non-outgoing variables', str(raises.exception))

    def test_case05_bogus_type_info(self):

        def foo(x):
            with objmode_context(z='float64[:]'):
                z = x + 1j
            return z
        x = np.array([1, 2, 3])
        cfoo = njit(foo)
        with self.assertRaises(TypeError) as raises:
            got = cfoo(x)
        self.assertIn("can't unbox array from PyObject into native value.  The object maybe of a different type", str(raises.exception))

    def test_case06_double_objmode(self):

        def foo(x):
            with objmode_context():
                with objmode_context():
                    print(x)
            return x
        with self.assertRaises(errors.TypingError) as raises:
            njit(foo)(123)
        pat = 'During: resolving callee type: type\\(ObjModeLiftedWith\\(<.*>\\)\\)'
        self.assertRegex(str(raises.exception), pat)

    def test_case07_mystery_key_error(self):

        def foo(x):
            with objmode_context():
                t = {'a': x}
                u = 3
            return (x, t, u)
        x = np.array([1, 2, 3])
        cfoo = njit(foo)
        with self.assertRaises(errors.TypingError) as raises:
            cfoo(x)
        exstr = str(raises.exception)
        self.assertIn("Missing type annotation on outgoing variable(s): ['t', 'u']", exstr)
        self.assertIn("Example code: with objmode(t='<add_type_as_string_here>')", exstr)

    def test_case08_raise_from_external(self):
        d = dict()

        def foo(x):
            for i in range(len(x)):
                with objmode_context():
                    k = str(i)
                    v = x[i]
                    d[k] = v
                    print(d['2'])
            return x
        x = np.array([1, 2, 3])
        cfoo = njit(foo)
        with self.assertRaises(KeyError) as raises:
            cfoo(x)
        self.assertEqual(str(raises.exception), "'2'")

    def test_case09_explicit_raise(self):

        def foo(x):
            with objmode_context():
                raise ValueError()
            return x
        x = np.array([1, 2, 3])
        cfoo = njit(foo)
        with self.assertRaises(errors.CompilerError) as raises:
            cfoo(x)
        self.assertIn('unsupported control flow due to raise statements inside with block', str(raises.exception))

    @expected_failure_for_list_arg
    def test_case10_mutate_across_contexts(self):

        def foo(x):
            with objmode_context(y='List(int64)'):
                y = [1, 2, 3]
            with objmode_context():
                y[2] = 10
            return y
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    def test_case10_mutate_array_across_contexts(self):

        def foo(x):
            with objmode_context(y='int64[:]'):
                y = np.asarray([1, 2, 3], dtype='int64')
            with objmode_context():
                y[2] = 10
            return y
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    def test_case11_define_function_in_context(self):

        def foo(x):
            with objmode_context():

                def bar(y):
                    return y + 1
            return x
        x = np.array([1, 2, 3])
        cfoo = njit(foo)
        with self.assertRaises(NameError) as raises:
            cfoo(x)
        self.assertIn("global name 'bar' is not defined", str(raises.exception))

    def test_case12_njit_inside_a_objmode_ctx(self):

        def bar(y):
            return y + 1

        def foo(x):
            with objmode_context(y='int64[:]'):
                y = njit(bar)(x).astype('int64')
            return x + y
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    def test_case14_return_direct_from_objmode_ctx(self):

        def foo(x):
            with objmode_context(x='int64[:]'):
                x += 1
                return x
        result = foo(np.array([1, 2, 3]))
        np.testing.assert_array_equal(np.array([2, 3, 4]), result)

    @unittest.expectedFailure
    def test_case15_close_over_objmode_ctx(self):

        def foo(x):
            j = 10

            def bar(x):
                with objmode_context(x='int64[:]'):
                    print(x)
                    return x + j
            return bar(x) + 2
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    @skip_unless_scipy
    def test_case16_scipy_call_in_objmode_ctx(self):
        from scipy import sparse as sp

        def foo(x):
            with objmode_context(k='int64'):
                print(x)
                spx = sp.csr_matrix(x)
                k = np.int64(spx[0, 0])
            return k
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    def test_case17_print_own_bytecode(self):
        import dis

        def foo(x):
            with objmode_context():
                dis.dis(foo)
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    @expected_failure_for_function_arg
    def test_case18_njitfunc_passed_to_objmode_ctx(self):

        def foo(func, x):
            with objmode_context():
                func(x[0])
        x = np.array([1, 2, 3])
        fn = njit(lambda z: z + 5)
        self.assert_equal_return_and_stdout(foo, fn, x)

    @expected_failure_py311
    @expected_failure_py312
    def test_case19_recursion(self):

        def foo(x):
            with objmode_context():
                if x == 0:
                    return 7
            ret = foo(x - 1)
            return ret
        with self.assertRaises((errors.TypingError, errors.CompilerError)) as raises:
            cfoo = njit(foo)
            cfoo(np.array([1, 2, 3]))
        msg = "Untyped global name 'foo'"
        self.assertIn(msg, str(raises.exception))

    @unittest.expectedFailure
    def test_case20_rng_works_ok(self):

        def foo(x):
            np.random.seed(0)
            y = np.random.rand()
            with objmode_context(z='float64'):
                z = np.random.rand()
            return x + z + y
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    def test_case21_rng_seed_works_ok(self):

        def foo(x):
            np.random.seed(0)
            y = np.random.rand()
            with objmode_context(z='float64'):
                np.random.seed(0)
                z = np.random.rand()
            return x + z + y
        x = np.array([1, 2, 3])
        self.assert_equal_return_and_stdout(foo, x)

    def test_example01(self):

        def bar(x):
            return np.asarray(list(reversed(x.tolist())))

        @njit
        def foo():
            x = np.arange(5)
            with objmode(y='intp[:]'):
                y = x + bar(x)
            return y
        self.assertPreciseEqual(foo(), foo.py_func())
        self.assertIs(objmode, objmode_context)

    def test_objmode_in_overload(self):

        def foo(s):
            pass

        @overload(foo)
        def foo_overload(s):

            def impl(s):
                with objmode(out='intp'):
                    out = s + 3
                return out
            return impl

        @numba.njit
        def f():
            return foo(1)
        self.assertEqual(f(), 1 + 3)

    def test_objmode_gv_variable(self):

        @njit
        def global_var():
            with objmode(val=gv_type):
                val = 12.3
            return val
        ret = global_var()
        self.assertIsInstance(ret, int)
        self.assertEqual(ret, 12)

    def test_objmode_gv_variable_error(self):

        @njit
        def global_var():
            with objmode(val=gv_type2):
                val = 123
            return val
        with self.assertRaisesRegex(errors.CompilerError, "Error handling objmode argument 'val'. Global 'gv_type2' is not defined."):
            global_var()

    def test_objmode_gv_mod_attr(self):

        @njit
        def modattr1():
            with objmode(val=types.intp):
                val = 12.3
            return val

        @njit
        def modattr2():
            with objmode(val=numba.types.intp):
                val = 12.3
            return val
        for fn in (modattr1, modattr2):
            with self.subTest(fn=str(fn)):
                ret = fn()
                self.assertIsInstance(ret, int)
                self.assertEqual(ret, 12)

    def test_objmode_gv_mod_attr_error(self):

        @njit
        def moderror():
            with objmode(val=types.THIS_DOES_NOT_EXIST):
                val = 12.3
            return val
        with self.assertRaisesRegex(errors.CompilerError, "Error handling objmode argument 'val'. Getattr cannot be resolved at compile-time"):
            moderror()

    def test_objmode_gv_mod_attr_error_multiple(self):

        @njit
        def moderror():
            with objmode(v1=types.intp, v2=types.THIS_DOES_NOT_EXIST, v3=types.float32):
                v1 = 12.3
                v2 = 12.3
                v3 = 12.3
            return val
        with self.assertRaisesRegex(errors.CompilerError, "Error handling objmode argument 'v2'. Getattr cannot be resolved at compile-time"):
            moderror()

    def test_objmode_closure_type_in_overload(self):

        def foo():
            pass

        @overload(foo)
        def foo_overload():
            shrubbery = types.float64[:]

            def impl():
                with objmode(out=shrubbery):
                    out = np.arange(10).astype(np.float64)
                return out
            return impl

        @njit
        def bar():
            return foo()
        self.assertPreciseEqual(bar(), np.arange(10).astype(np.float64))

    def test_objmode_closure_type_in_overload_error(self):

        def foo():
            pass

        @overload(foo)
        def foo_overload():
            shrubbery = types.float64[:]

            def impl():
                with objmode(out=shrubbery):
                    out = np.arange(10).astype(np.float64)
                return out
            del shrubbery
            return impl

        @njit
        def bar():
            return foo()
        with self.assertRaisesRegex(errors.TypingError, "Error handling objmode argument 'out'. Freevar 'shrubbery' is not defined"):
            bar()

    def test_objmode_invalid_use(self):

        @njit
        def moderror():
            with objmode(bad=1 + 1):
                out = 1
            return val
        with self.assertRaisesRegex(errors.CompilerError, "Error handling objmode argument 'bad'. The value must be a compile-time constant either as a non-local variable or a getattr expression that refers to a Numba type."):
            moderror()

    def test_objmode_multi_type_args(self):
        array_ty = types.int32[:]

        @njit
        def foo():
            with objmode(t1='float64', t2=gv_type, t3=array_ty):
                t1 = 793856.5
                t2 = t1
                t3 = np.arange(5).astype(np.int32)
            return (t1, t2, t3)
        t1, t2, t3 = foo()
        self.assertPreciseEqual(t1, 793856.5)
        self.assertPreciseEqual(t2, 793856)
        self.assertPreciseEqual(t3, np.arange(5).astype(np.int32))

    def test_objmode_jitclass(self):
        spec = [('value', types.int32), ('array', types.float32[:])]

        @jitclass(spec)
        class Bag(object):

            def __init__(self, value):
                self.value = value
                self.array = np.zeros(value, dtype=np.float32)

            @property
            def size(self):
                return self.array.size

            def increment(self, val):
                for i in range(self.size):
                    self.array[i] += val
                return self.array

            @staticmethod
            def add(x, y):
                return x + y
        n = 21
        mybag = Bag(n)

        def foo():
            pass

        @overload(foo)
        def foo_overload():
            shrubbery = mybag._numba_type_

            def impl():
                with objmode(out=shrubbery):
                    out = Bag(123)
                    out.increment(3)
                return out
            return impl

        @njit
        def bar():
            return foo()
        z = bar()
        self.assertIsInstance(z, Bag)
        self.assertEqual(z.add(2, 3), 2 + 3)
        exp_array = np.zeros(123, dtype=np.float32) + 3
        self.assertPreciseEqual(z.array, exp_array)

    @staticmethod
    def case_objmode_cache(x):
        with objmode(output='float64'):
            output = x / 10
        return output

    def test_objmode_reflected_list(self):
        ret_type = typeof([1, 2, 3, 4, 5])

        @njit
        def test2():
            with objmode(out=ret_type):
                out = [1, 2, 3, 4, 5]
            return out
        with self.assertRaises(errors.CompilerError) as raises:
            test2()
        self.assertRegex(str(raises.exception), "Objmode context failed. Argument 'out' is declared as an unsupported type: reflected list\\(int(32|64)\\)<iv=None>. Reflected types are not supported.")

    def test_objmode_reflected_set(self):
        ret_type = typeof({1, 2, 3, 4, 5})

        @njit
        def test2():
            with objmode(result=ret_type):
                result = {1, 2, 3, 4, 5}
            return result
        with self.assertRaises(errors.CompilerError) as raises:
            test2()
        self.assertRegex(str(raises.exception), "Objmode context failed. Argument 'result' is declared as an unsupported type: reflected set\\(int(32|64)\\). Reflected types are not supported.")

    def test_objmode_typed_dict(self):
        ret_type = types.DictType(types.unicode_type, types.int64)

        @njit
        def test4():
            with objmode(res=ret_type):
                res = {'A': 1, 'B': 2}
            return res
        with self.assertRaises(TypeError) as raises:
            test4()
        self.assertIn("can't unbox a <class 'dict'> as a <class 'numba.typed.typeddict.Dict'>", str(raises.exception))

    def test_objmode_typed_list(self):
        ret_type = types.ListType(types.int64)

        @njit
        def test4():
            with objmode(res=ret_type):
                res = [1, 2]
            return res
        with self.assertRaises(TypeError) as raises:
            test4()
        self.assertRegex(str(raises.exception), "can't unbox a <class 'list'> as a (<class ')?numba.typed.typedlist.List('>)?")

    def test_objmode_use_of_view(self):

        @njit
        def foo(x):
            with numba.objmode(y='int64[::1]'):
                y = x.view('int64')
            return y
        a = np.ones(1, np.int64).view('float64')
        expected = foo.py_func(a)
        got = foo(a)
        self.assertPreciseEqual(expected, got)