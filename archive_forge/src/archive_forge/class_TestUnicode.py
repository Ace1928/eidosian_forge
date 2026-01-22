from itertools import product
from itertools import permutations
from numba import njit, typeof
from numba.core import types
import unittest
from numba.tests.support import (TestCase, no_pyobj_flags, MemoryLeakMixin)
from numba.core.errors import TypingError, UnsupportedError
from numba.cpython.unicode import _MAX_UNICODE
from numba.core.types.functions import _header_lead
from numba.extending import overload
class TestUnicode(BaseTest):

    def test_literal(self):
        pyfunc = literal_usecase
        cfunc = njit(literal_usecase)
        self.assertPreciseEqual(pyfunc(), cfunc())

    def test_passthrough(self, flags=no_pyobj_flags):
        pyfunc = passthrough_usecase
        cfunc = njit(pyfunc)
        for s in UNICODE_EXAMPLES:
            self.assertEqual(pyfunc(s), cfunc(s))

    def test_eq(self, flags=no_pyobj_flags):
        pyfunc = eq_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for b in reversed(UNICODE_EXAMPLES):
                self.assertEqual(pyfunc(a, b), cfunc(a, b), '%s, %s' % (a, b))
                self.assertEqual(pyfunc(a, 1), cfunc(a, 1), '%s, %s' % (a, 1))
                self.assertEqual(pyfunc(1, b), cfunc(1, b), '%s, %s' % (1, b))

    def test_eq_optional(self):

        @njit
        def foo(pred1, pred2):
            if pred1 > 0:
                resolved1 = 'concrete'
            else:
                resolved1 = None
            if pred2 < 0:
                resolved2 = 'concrete'
            else:
                resolved2 = None
            if resolved1 == resolved2:
                return 10
            else:
                return 20
        for p1, p2 in product(*((-1, 1),) * 2):
            self.assertEqual(foo(p1, p2), foo.py_func(p1, p2))

    def _check_ordering_op(self, usecase):
        pyfunc = usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_ORDERING_EXAMPLES:
            self.assertEqual(pyfunc(a, a), cfunc(a, a), '%s: "%s", "%s"' % (usecase.__name__, a, a))
        for a, b in permutations(UNICODE_ORDERING_EXAMPLES, r=2):
            self.assertEqual(pyfunc(a, b), cfunc(a, b), '%s: "%s", "%s"' % (usecase.__name__, a, b))
            self.assertEqual(pyfunc(b, a), cfunc(b, a), '%s: "%s", "%s"' % (usecase.__name__, b, a))

    def test_lt(self, flags=no_pyobj_flags):
        self._check_ordering_op(lt_usecase)

    def test_le(self, flags=no_pyobj_flags):
        self._check_ordering_op(le_usecase)

    def test_gt(self, flags=no_pyobj_flags):
        self._check_ordering_op(gt_usecase)

    def test_ge(self, flags=no_pyobj_flags):
        self._check_ordering_op(ge_usecase)

    def test_len(self, flags=no_pyobj_flags):
        pyfunc = len_usecase
        cfunc = njit(pyfunc)
        for s in UNICODE_EXAMPLES:
            self.assertEqual(pyfunc(s), cfunc(s))

    def test_bool(self, flags=no_pyobj_flags):
        pyfunc = bool_usecase
        cfunc = njit(pyfunc)
        for s in UNICODE_EXAMPLES:
            self.assertEqual(pyfunc(s), cfunc(s))

    def test_expandtabs(self):
        pyfunc = expandtabs_usecase
        cfunc = njit(pyfunc)
        cases = ['', '\t', 't\tt\t', 'a\t', '\t⚡', 'a\tbc\nab\tc', '🐍\t⚡', '🐍⚡\n\t\t🐍\t', 'ab\rab\t\t\tab\r\n\ta']
        msg = 'Results of "{}".expandtabs() must be equal'
        for s in cases:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_expandtabs_with_tabsize(self):
        fns = [njit(expandtabs_with_tabsize_usecase), njit(expandtabs_with_tabsize_kwarg_usecase)]
        messages = ['Results of "{}".expandtabs({}) must be equal', 'Results of "{}".expandtabs(tabsize={}) must be equal']
        cases = ['', '\t', 't\tt\t', 'a\t', '\t⚡', 'a\tbc\nab\tc', '🐍\t⚡', '🐍⚡\n\t\t🐍\t', 'ab\rab\t\t\tab\r\n\ta']
        for s in cases:
            for tabsize in range(-1, 10):
                for fn, msg in zip(fns, messages):
                    self.assertEqual(fn.py_func(s, tabsize), fn(s, tabsize), msg=msg.format(s, tabsize))

    def test_expandtabs_exception_noninteger_tabsize(self):
        pyfunc = expandtabs_with_tabsize_usecase
        cfunc = njit(pyfunc)
        accepted_types = (types.Integer, int)
        with self.assertRaises(TypingError) as raises:
            cfunc('\t', 2.4)
        msg = '"tabsize" must be {}, not float'.format(accepted_types)
        self.assertIn(msg, str(raises.exception))

    def test_startswith_default(self):
        pyfunc = startswith_usecase
        cfunc = njit(pyfunc)
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = ['he', 'hello', 'helloworld', 'ello', '', 'lowo', 'lo', 'he', 'lo', 'o']
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for prefix in cpython_subs + default_subs + extra_subs:
                self.assertEqual(pyfunc(s, prefix), cfunc(s, prefix))

    def test_startswith_with_start(self):
        pyfunc = startswith_with_start_only_usecase
        cfunc = njit(pyfunc)
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = ['he', 'hello', 'helloworld', 'ello', '', 'lowo', 'lo', 'he', 'lo', 'o']
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for prefix in cpython_subs + default_subs + extra_subs:
                for start in list(range(-20, 20)) + [None]:
                    self.assertEqual(pyfunc(s, prefix, start), cfunc(s, prefix, start))

    def test_startswith_with_start_end(self):
        pyfunc = startswith_with_start_end_usecase
        cfunc = njit(pyfunc)
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = ['he', 'hello', 'helloworld', 'ello', '', 'lowo', 'lo', 'he', 'lo', 'o']
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for prefix in cpython_subs + default_subs + extra_subs:
                for start in list(range(-20, 20)) + [None]:
                    for end in list(range(-20, 20)) + [None]:
                        self.assertEqual(pyfunc(s, prefix, start, end), cfunc(s, prefix, start, end))

    def test_startswith_exception_invalid_args(self):
        msg_invalid_prefix = "The arg 'prefix' should be a string or a tuple of strings"
        with self.assertRaisesRegex(TypingError, msg_invalid_prefix):
            cfunc = njit(startswith_usecase)
            cfunc('hello', (1, 'he'))
        msg_invalid_start = "When specified, the arg 'start' must be an Integer or None"
        with self.assertRaisesRegex(TypingError, msg_invalid_start):
            cfunc = njit(startswith_with_start_only_usecase)
            cfunc('hello', 'he', 'invalid start')
        msg_invalid_end = "When specified, the arg 'end' must be an Integer or None"
        with self.assertRaisesRegex(TypingError, msg_invalid_end):
            cfunc = njit(startswith_with_start_end_usecase)
            cfunc('hello', 'he', 0, 'invalid end')

    def test_startswith_tuple(self):
        pyfunc = startswith_usecase
        cfunc = njit(pyfunc)
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = ['he', 'hello', 'helloworld', 'ello', '', 'lowo', 'lo', 'he', 'lo', 'o']
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for sub_str in cpython_subs + default_subs + extra_subs:
                prefix = (sub_str, 'lo')
                self.assertEqual(pyfunc(s, prefix), cfunc(s, prefix))

    def test_startswith_tuple_args(self):
        pyfunc = startswith_with_start_end_usecase
        cfunc = njit(pyfunc)
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = ['he', 'hello', 'helloworld', 'ello', '', 'lowo', 'lo', 'he', 'lo', 'o']
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for sub_str in cpython_subs + default_subs + extra_subs:
                for start in list(range(-20, 20)) + [None]:
                    for end in list(range(-20, 20)) + [None]:
                        prefix = (sub_str, 'lo')
                        self.assertEqual(pyfunc(s, prefix, start, end), cfunc(s, prefix, start, end))

    def test_endswith_default(self):
        pyfunc = endswith_usecase
        cfunc = njit(pyfunc)
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = ['he', 'hello', 'helloworld', 'ello', '', 'lowo', 'lo', 'he', 'lo', 'o']
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for sub_str in cpython_subs + default_subs + extra_subs:
                msg = 'Results "{}".endswith("{}") must be equal'
                self.assertEqual(pyfunc(s, sub_str), cfunc(s, sub_str), msg=msg.format(s, sub_str))

    def test_endswith_with_start(self):
        pyfunc = endswith_with_start_only_usecase
        cfunc = njit(pyfunc)
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = ['he', 'hello', 'helloworld', 'ello', '', 'lowo', 'lo', 'he', 'lo', 'o']
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for sub_str in cpython_subs + default_subs + extra_subs:
                for start in list(range(-20, 20)) + [None]:
                    msg = 'Results "{}".endswith("{}", {}) must be equal'
                    self.assertEqual(pyfunc(s, sub_str, start), cfunc(s, sub_str, start), msg=msg.format(s, sub_str, start))

    def test_endswith_with_start_end(self):
        pyfunc = endswith_with_start_end_usecase
        cfunc = njit(pyfunc)
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = ['he', 'hello', 'helloworld', 'ello', '', 'lowo', 'lo', 'he', 'lo', 'o']
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for sub_str in cpython_subs + default_subs + extra_subs:
                for start in list(range(-20, 20)) + [None]:
                    for end in list(range(-20, 20)) + [None]:
                        msg = 'Results "{}".endswith("{}", {}, {})                               must be equal'
                        self.assertEqual(pyfunc(s, sub_str, start, end), cfunc(s, sub_str, start, end), msg=msg.format(s, sub_str, start, end))

    def test_endswith_tuple(self):
        pyfunc = endswith_usecase
        cfunc = njit(pyfunc)
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = ['he', 'hello', 'helloworld', 'ello', '', 'lowo', 'lo', 'he', 'lo', 'o']
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for sub_str in cpython_subs + default_subs + extra_subs:
                msg = 'Results "{}".endswith({}) must be equal'
                tuple_subs = (sub_str, 'lo')
                self.assertEqual(pyfunc(s, tuple_subs), cfunc(s, tuple_subs), msg=msg.format(s, tuple_subs))

    def test_endswith_tuple_args(self):
        pyfunc = endswith_with_start_end_usecase
        cfunc = njit(pyfunc)
        cpython_str = ['hello', 'helloworld', '']
        cpython_subs = ['he', 'hello', 'helloworld', 'ello', '', 'lowo', 'lo', 'he', 'lo', 'o']
        extra_subs = ['hellohellohello', ' ']
        for s in cpython_str + UNICODE_EXAMPLES:
            default_subs = ['', 'x', s[:-2], s[3:], s, s + s]
            for sub_str in cpython_subs + default_subs + extra_subs:
                for start in list(range(-20, 20)) + [None]:
                    for end in list(range(-20, 20)) + [None]:
                        msg = 'Results "{}".endswith("{}", {}, {})                               must be equal'
                        tuple_subs = (sub_str, 'lo')
                        self.assertEqual(pyfunc(s, tuple_subs, start, end), cfunc(s, tuple_subs, start, end), msg=msg.format(s, tuple_subs, start, end))

    def test_in(self, flags=no_pyobj_flags):
        pyfunc = in_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            extras = ['', 'xx', a[::-1], a[:-2], a[3:], a, a + a]
            for substr in extras:
                self.assertEqual(pyfunc(substr, a), cfunc(substr, a), "'%s' in '%s'?" % (substr, a))

    def test_partition_exception_invalid_sep(self):
        self.disable_leak_check()
        pyfunc = partition_usecase
        cfunc = njit(pyfunc)
        for func in [pyfunc, cfunc]:
            with self.assertRaises(ValueError) as raises:
                func('a', '')
            self.assertIn('empty separator', str(raises.exception))
        accepted_types = (types.UnicodeType, types.UnicodeCharSeq)
        with self.assertRaises(TypingError) as raises:
            cfunc('a', None)
        msg = '"sep" must be {}, not none'.format(accepted_types)
        self.assertIn(msg, str(raises.exception))

    def test_partition(self):
        pyfunc = partition_usecase
        cfunc = njit(pyfunc)
        CASES = [('', '⚡'), ('abcabc', '⚡'), ('🐍⚡', '⚡'), ('🐍⚡🐍', '⚡'), ('abababa', 'a'), ('abababa', 'b'), ('abababa', 'c'), ('abababa', 'ab'), ('abababa', 'aba')]
        msg = 'Results of "{}".partition("{}") must be equal'
        for s, sep in CASES:
            self.assertEqual(pyfunc(s, sep), cfunc(s, sep), msg=msg.format(s, sep))

    def test_find(self, flags=no_pyobj_flags):
        pyfunc = find_usecase
        cfunc = njit(pyfunc)
        default_subs = [(s, ['', 'xx', s[:-2], s[3:], s]) for s in UNICODE_EXAMPLES]
        cpython_subs = [('a' * 100 + 'Ă', ['Ă', 'ȁ', 'Ġ', 'Ƞ']), ('a' * 100 + '\U00100304', ['\U00100304', '\U00100204', '\U00102004']), ('Ă' * 100 + 'a', ['a']), ('\U00100304' * 100 + 'a', ['a']), ('\U00100304' * 100 + 'Ă', ['Ă']), ('a' * 100, ['Ă', '\U00100304', 'aĂ', 'a\U00100304']), ('Ă' * 100, ['\U00100304', 'Ă\U00100304']), ('Ă' * 100 + 'a_', ['a_']), ('\U00100304' * 100 + 'a_', ['a_']), ('\U00100304' * 100 + 'Ă_', ['Ă_'])]
        for s, subs in default_subs + cpython_subs:
            for sub_str in subs:
                msg = 'Results "{}".find("{}") must be equal'
                self.assertEqual(pyfunc(s, sub_str), cfunc(s, sub_str), msg=msg.format(s, sub_str))

    def test_find_with_start_only(self):
        pyfunc = find_with_start_only_usecase
        cfunc = njit(pyfunc)
        for s in UNICODE_EXAMPLES:
            for sub_str in ['', 'xx', s[:-2], s[3:], s]:
                for start in list(range(-20, 20)) + [None]:
                    msg = 'Results "{}".find("{}", {}) must be equal'
                    self.assertEqual(pyfunc(s, sub_str, start), cfunc(s, sub_str, start), msg=msg.format(s, sub_str, start))

    def test_find_with_start_end(self):
        pyfunc = find_with_start_end_usecase
        cfunc = njit(pyfunc)
        starts = ends = list(range(-20, 20)) + [None]
        for s in UNICODE_EXAMPLES:
            for sub_str in ['', 'xx', s[:-2], s[3:], s]:
                for start, end in product(starts, ends):
                    msg = 'Results of "{}".find("{}", {}, {}) must be equal'
                    self.assertEqual(pyfunc(s, sub_str, start, end), cfunc(s, sub_str, start, end), msg=msg.format(s, sub_str, start, end))

    def test_find_exception_noninteger_start_end(self):
        pyfunc = find_with_start_end_usecase
        cfunc = njit(pyfunc)
        accepted = (types.Integer, types.NoneType)
        for start, end, name in [(0.1, 5, 'start'), (0, 0.5, 'end')]:
            with self.assertRaises(TypingError) as raises:
                cfunc('ascii', 'sci', start, end)
            msg = '"{}" must be {}, not float'.format(name, accepted)
            self.assertIn(msg, str(raises.exception))

    def test_rpartition_exception_invalid_sep(self):
        self.disable_leak_check()
        pyfunc = rpartition_usecase
        cfunc = njit(pyfunc)
        for func in [pyfunc, cfunc]:
            with self.assertRaises(ValueError) as raises:
                func('a', '')
            self.assertIn('empty separator', str(raises.exception))
        accepted_types = (types.UnicodeType, types.UnicodeCharSeq)
        with self.assertRaises(TypingError) as raises:
            cfunc('a', None)
        msg = '"sep" must be {}, not none'.format(accepted_types)
        self.assertIn(msg, str(raises.exception))

    def test_rpartition(self):
        pyfunc = rpartition_usecase
        cfunc = njit(pyfunc)
        CASES = [('', '⚡'), ('abcabc', '⚡'), ('🐍⚡', '⚡'), ('🐍⚡🐍', '⚡'), ('abababa', 'a'), ('abababa', 'b'), ('abababa', 'c'), ('abababa', 'ab'), ('abababa', 'aba')]
        msg = 'Results of "{}".rpartition("{}") must be equal'
        for s, sep in CASES:
            self.assertEqual(pyfunc(s, sep), cfunc(s, sep), msg=msg.format(s, sep))

    def test_count(self):
        pyfunc = count_usecase
        cfunc = njit(pyfunc)
        error_msg = "'{0}'.py_count('{1}') = {2}\n'{0}'.c_count('{1}') = {3}"
        for s, sub in UNICODE_COUNT_EXAMPLES:
            py_result = pyfunc(s, sub)
            c_result = cfunc(s, sub)
            self.assertEqual(py_result, c_result, error_msg.format(s, sub, py_result, c_result))

    def test_count_with_start(self):
        pyfunc = count_with_start_usecase
        cfunc = njit(pyfunc)
        error_msg = '%s\n%s' % ("'{0}'.py_count('{1}', {2}) = {3}", "'{0}'.c_count('{1}', {2}) = {4}")
        for s, sub in UNICODE_COUNT_EXAMPLES:
            for i in range(-18, 18):
                py_result = pyfunc(s, sub, i)
                c_result = cfunc(s, sub, i)
                self.assertEqual(py_result, c_result, error_msg.format(s, sub, i, py_result, c_result))
            py_result = pyfunc(s, sub, None)
            c_result = cfunc(s, sub, None)
            self.assertEqual(py_result, c_result, error_msg.format(s, sub, None, py_result, c_result))

    def test_count_with_start_end(self):
        pyfunc = count_with_start_end_usecase
        cfunc = njit(pyfunc)
        error_msg = '%s\n%s' % ("'{0}'.py_count('{1}', {2}, {3}) = {4}", "'{0}'.c_count('{1}', {2}, {3}) = {5}")
        for s, sub in UNICODE_COUNT_EXAMPLES:
            for i, j in product(range(-18, 18), (-18, 18)):
                py_result = pyfunc(s, sub, i, j)
                c_result = cfunc(s, sub, i, j)
                self.assertEqual(py_result, c_result, error_msg.format(s, sub, i, j, py_result, c_result))
            for j in range(-18, 18):
                py_result = pyfunc(s, sub, None, j)
                c_result = cfunc(s, sub, None, j)
                self.assertEqual(py_result, c_result, error_msg.format(s, sub, None, j, py_result, c_result))
            py_result = pyfunc(s, sub, None, None)
            c_result = cfunc(s, sub, None, None)
            self.assertEqual(py_result, c_result, error_msg.format(s, sub, None, None, py_result, c_result))

    def test_count_arg_type_check(self):
        cfunc = njit(count_with_start_end_usecase)
        with self.assertRaises(TypingError) as raises:
            cfunc('ascii', 'c', 1, 0.5)
        self.assertIn('The slice indices must be an Integer or None', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc('ascii', 'c', 1.2, 7)
        self.assertIn('The slice indices must be an Integer or None', str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc('ascii', 12, 1, 7)
        self.assertIn('The substring must be a UnicodeType, not', str(raises.exception))

    def test_count_optional_arg_type_check(self):
        pyfunc = count_with_start_end_usecase

        def try_compile_bad_optional(*args):
            bad_sig = types.int64(types.unicode_type, types.unicode_type, types.Optional(types.float64), types.Optional(types.float64))
            njit([bad_sig])(pyfunc)
        with self.assertRaises(TypingError) as raises:
            try_compile_bad_optional('tú quis?', 'tú', 1.1, 1.1)
        self.assertIn('The slice indices must be an Integer or None', str(raises.exception))
        error_msg = '%s\n%s' % ("'{0}'.py_count('{1}', {2}, {3}) = {4}", "'{0}'.c_count_op('{1}', {2}, {3}) = {5}")
        sig_optional = types.int64(types.unicode_type, types.unicode_type, types.Optional(types.int64), types.Optional(types.int64))
        cfunc_optional = njit([sig_optional])(pyfunc)
        py_result = pyfunc('tú quis?', 'tú', 0, 8)
        c_result = cfunc_optional('tú quis?', 'tú', 0, 8)
        self.assertEqual(py_result, c_result, error_msg.format('tú quis?', 'tú', 0, 8, py_result, c_result))

    def test_rfind(self):
        pyfunc = rfind_usecase
        cfunc = njit(pyfunc)
        default_subs = [(s, ['', 'xx', s[:-2], s[3:], s]) for s in UNICODE_EXAMPLES]
        cpython_subs = [('Ă' + 'a' * 100, ['Ă', 'ȁ', 'Ġ', 'Ƞ']), ('\U00100304' + 'a' * 100, ['\U00100304', '\U00100204', '\U00102004']), ('abcdefghiabc', ['abc', '']), ('a' + 'Ă' * 100, ['a']), ('a' + '\U00100304' * 100, ['a']), ('Ă' + '\U00100304' * 100, ['Ă']), ('a' * 100, ['Ă', '\U00100304', 'Ăa', '\U00100304a']), ('Ă' * 100, ['\U00100304', '\U00100304Ă']), ('_a' + 'Ă' * 100, ['_a']), ('_a' + '\U00100304' * 100, ['_a']), ('_Ă' + '\U00100304' * 100, ['_Ă'])]
        for s, subs in default_subs + cpython_subs:
            for sub_str in subs:
                msg = 'Results "{}".rfind("{}") must be equal'
                self.assertEqual(pyfunc(s, sub_str), cfunc(s, sub_str), msg=msg.format(s, sub_str))

    def test_rfind_with_start_only(self):
        pyfunc = rfind_with_start_only_usecase
        cfunc = njit(pyfunc)
        for s in UNICODE_EXAMPLES:
            for sub_str in ['', 'xx', s[:-2], s[3:], s]:
                for start in list(range(-20, 20)) + [None]:
                    msg = 'Results "{}".rfind("{}", {}) must be equal'
                    self.assertEqual(pyfunc(s, sub_str, start), cfunc(s, sub_str, start), msg=msg.format(s, sub_str, start))

    def test_rfind_with_start_end(self):
        pyfunc = rfind_with_start_end_usecase
        cfunc = njit(pyfunc)
        starts = list(range(-20, 20)) + [None]
        ends = list(range(-20, 20)) + [None]
        for s in UNICODE_EXAMPLES:
            for sub_str in ['', 'xx', s[:-2], s[3:], s]:
                for start, end in product(starts, ends):
                    msg = 'Results of "{}".rfind("{}", {}, {}) must be equal'
                    self.assertEqual(pyfunc(s, sub_str, start, end), cfunc(s, sub_str, start, end), msg=msg.format(s, sub_str, start, end))

    def test_rfind_wrong_substr(self):
        cfunc = njit(rfind_usecase)
        for s in UNICODE_EXAMPLES:
            for sub_str in [None, 1, False]:
                with self.assertRaises(TypingError) as raises:
                    cfunc(s, sub_str)
                msg = 'must be {}'.format(types.UnicodeType)
                self.assertIn(msg, str(raises.exception))

    def test_rfind_wrong_start_end(self):
        cfunc = njit(rfind_with_start_end_usecase)
        accepted_types = (types.Integer, types.NoneType)
        for s in UNICODE_EXAMPLES:
            for sub_str in ['', 'xx', s[:-2], s[3:], s]:
                for start, end in product([0.1, False], [-1, 1]):
                    with self.assertRaises(TypingError) as raises:
                        cfunc(s, sub_str, start, end)
                    msg = '"start" must be {}'.format(accepted_types)
                    self.assertIn(msg, str(raises.exception))
                for start, end in product([-1, 1], [-0.1, True]):
                    with self.assertRaises(TypingError) as raises:
                        cfunc(s, sub_str, start, end)
                    msg = '"end" must be {}'.format(accepted_types)
                    self.assertIn(msg, str(raises.exception))

    def test_rfind_wrong_start_end_optional(self):
        s = UNICODE_EXAMPLES[0]
        sub_str = s[1:-1]
        accepted_types = (types.Integer, types.NoneType)
        msg = 'must be {}'.format(accepted_types)

        def try_compile_wrong_start_optional(*args):
            wrong_sig_optional = types.int64(types.unicode_type, types.unicode_type, types.Optional(types.float64), types.Optional(types.intp))
            njit([wrong_sig_optional])(rfind_with_start_end_usecase)
        with self.assertRaises(TypingError) as raises:
            try_compile_wrong_start_optional(s, sub_str, 0.1, 1)
        self.assertIn(msg, str(raises.exception))

        def try_compile_wrong_end_optional(*args):
            wrong_sig_optional = types.int64(types.unicode_type, types.unicode_type, types.Optional(types.intp), types.Optional(types.float64))
            njit([wrong_sig_optional])(rfind_with_start_end_usecase)
        with self.assertRaises(TypingError) as raises:
            try_compile_wrong_end_optional(s, sub_str, 1, 0.1)
        self.assertIn(msg, str(raises.exception))

    def test_rindex(self):
        pyfunc = rindex_usecase
        cfunc = njit(pyfunc)
        default_subs = [(s, ['', s[:-2], s[3:], s]) for s in UNICODE_EXAMPLES]
        cpython_subs = [('abcdefghiabc', ['', 'def', 'abc']), ('a' + 'Ă' * 100, ['a']), ('a' + '\U00100304' * 100, ['a']), ('Ă' + '\U00100304' * 100, ['Ă']), ('_a' + 'Ă' * 100, ['_a']), ('_a' + '\U00100304' * 100, ['_a']), ('_Ă' + '\U00100304' * 100, ['_Ă'])]
        for s, subs in default_subs + cpython_subs:
            for sub_str in subs:
                msg = 'Results "{}".rindex("{}") must be equal'
                self.assertEqual(pyfunc(s, sub_str), cfunc(s, sub_str), msg=msg.format(s, sub_str))

    def test_index(self):
        pyfunc = index_usecase
        cfunc = njit(pyfunc)
        default_subs = [(s, ['', s[:-2], s[3:], s]) for s in UNICODE_EXAMPLES]
        cpython_subs = [('abcdefghiabc', ['', 'def', 'abc']), ('Ă' * 100 + 'a', ['a']), ('\U00100304' * 100 + 'a', ['a']), ('\U00100304' * 100 + 'Ă', ['Ă']), ('Ă' * 100 + 'a_', ['a_']), ('\U00100304' * 100 + 'a_', ['a_']), ('\U00100304' * 100 + 'Ă_', ['Ă_'])]
        for s, subs in default_subs + cpython_subs:
            for sub_str in subs:
                msg = 'Results "{}".index("{}") must be equal'
                self.assertEqual(pyfunc(s, sub_str), cfunc(s, sub_str), msg=msg.format(s, sub_str))

    def test_index_rindex_with_start_only(self):
        pyfuncs = [index_with_start_only_usecase, rindex_with_start_only_usecase]
        messages = ['Results "{}".index("{}", {}) must be equal', 'Results "{}".rindex("{}", {}) must be equal']
        unicode_examples = ['ascii', '12345', '1234567890', '¡Y tú quién te crees?', '大处着眼，小处着手。']
        for pyfunc, msg in zip(pyfuncs, messages):
            cfunc = njit(pyfunc)
            for s in unicode_examples:
                l = len(s)
                cases = [('', list(range(-10, l + 1))), (s[:-2], [0] + list(range(-10, 1 - l))), (s[3:], list(range(4)) + list(range(-10, 4 - l))), (s, [0] + list(range(-10, 1 - l)))]
                for sub_str, starts in cases:
                    for start in starts + [None]:
                        self.assertEqual(pyfunc(s, sub_str, start), cfunc(s, sub_str, start), msg=msg.format(s, sub_str, start))

    def test_index_rindex_with_start_end(self):
        pyfuncs = [index_with_start_end_usecase, rindex_with_start_end_usecase]
        messages = ['Results of "{}".index("{}", {}, {}) must be equal', 'Results of "{}".rindex("{}", {}, {}) must be equal']
        unicode_examples = ['ascii', '12345', '1234567890', '¡Y tú quién te crees?', '大处着眼，小处着手。']
        for pyfunc, msg in zip(pyfuncs, messages):
            cfunc = njit(pyfunc)
            for s in unicode_examples:
                l = len(s)
                cases = [('', list(range(-10, l + 1)), list(range(l, 10))), (s[:-2], [0] + list(range(-10, 1 - l)), [-2, -1] + list(range(l - 2, 10))), (s[3:], list(range(4)) + list(range(-10, -1)), list(range(l, 10))), (s, [0] + list(range(-10, 1 - l)), list(range(l, 10)))]
                for sub_str, starts, ends in cases:
                    for start, end in product(starts + [None], ends):
                        self.assertEqual(pyfunc(s, sub_str, start, end), cfunc(s, sub_str, start, end), msg=msg.format(s, sub_str, start, end))

    def test_index_rindex_exception_substring_not_found(self):
        self.disable_leak_check()
        unicode_examples = ['ascii', '12345', '1234567890', '¡Y tú quién te crees?', '大处着眼，小处着手。']
        pyfuncs = [index_with_start_end_usecase, rindex_with_start_end_usecase]
        for pyfunc in pyfuncs:
            cfunc = njit(pyfunc)
            for s in unicode_examples:
                l = len(s)
                cases = [('', list(range(l + 1, 10)), [l]), (s[:-2], [0], list(range(l - 2))), (s[3:], list(range(4, 10)), [l]), (s, [None], list(range(l)))]
                for sub_str, starts, ends in cases:
                    for start, end in product(starts, ends):
                        for func in [pyfunc, cfunc]:
                            with self.assertRaises(ValueError) as raises:
                                func(s, sub_str, start, end)
                            msg = 'substring not found'
                            self.assertIn(msg, str(raises.exception))

    def test_index_rindex_exception_noninteger_start_end(self):
        accepted = (types.Integer, types.NoneType)
        pyfuncs = [index_with_start_end_usecase, rindex_with_start_end_usecase]
        for pyfunc in pyfuncs:
            cfunc = njit(pyfunc)
            for start, end, name in [(0.1, 5, 'start'), (0, 0.5, 'end')]:
                with self.assertRaises(TypingError) as raises:
                    cfunc('ascii', 'sci', start, end)
                msg = '"{}" must be {}, not float'.format(name, accepted)
                self.assertIn(msg, str(raises.exception))

    def test_getitem(self):
        pyfunc = getitem_usecase
        cfunc = njit(pyfunc)
        for s in UNICODE_EXAMPLES:
            for i in range(-len(s), len(s)):
                self.assertEqual(pyfunc(s, i), cfunc(s, i), "'%s'[%d]?" % (s, i))

    def test_getitem_scalar_kind(self):
        pyfunc = getitem_check_kind_usecase
        cfunc = njit(pyfunc)
        samples = ['aሴ', '¡着']
        for s in samples:
            for i in range(-len(s), len(s)):
                self.assertEqual(pyfunc(s, i), cfunc(s, i), "'%s'[%d]?" % (s, i))

    def test_getitem_error(self):
        self.disable_leak_check()
        pyfunc = getitem_usecase
        cfunc = njit(pyfunc)
        for s in UNICODE_EXAMPLES:
            with self.assertRaises(IndexError) as raises:
                pyfunc(s, len(s))
            self.assertIn('string index out of range', str(raises.exception))
            with self.assertRaises(IndexError) as raises:
                cfunc(s, len(s))
            self.assertIn('string index out of range', str(raises.exception))

    def test_slice2(self):
        pyfunc = getitem_usecase
        cfunc = njit(pyfunc)
        for s in UNICODE_EXAMPLES:
            for i in list(range(-len(s), len(s))):
                for j in list(range(-len(s), len(s))):
                    sl = slice(i, j)
                    self.assertEqual(pyfunc(s, sl), cfunc(s, sl), "'%s'[%d:%d]?" % (s, i, j))

    def test_slice2_error(self):
        pyfunc = getitem_usecase
        cfunc = njit(pyfunc)
        for s in UNICODE_EXAMPLES:
            for i in [-2, -1, len(s), len(s) + 1]:
                for j in [-2, -1, len(s), len(s) + 1]:
                    sl = slice(i, j)
                    self.assertEqual(pyfunc(s, sl), cfunc(s, sl), "'%s'[%d:%d]?" % (s, i, j))

    def test_getitem_slice2_kind(self):
        pyfunc = getitem_check_kind_usecase
        cfunc = njit(pyfunc)
        samples = ['abcሴሴ', '¡¡¡着着着']
        for s in samples:
            for i in [-2, -1, 0, 1, 2, len(s), len(s) + 1]:
                for j in [-2, -1, 0, 1, 2, len(s), len(s) + 1]:
                    sl = slice(i, j)
                    self.assertEqual(pyfunc(s, sl), cfunc(s, sl), "'%s'[%d:%d]?" % (s, i, j))

    def test_slice3(self):
        pyfunc = getitem_usecase
        cfunc = njit(pyfunc)
        for s in UNICODE_EXAMPLES:
            for i in range(-len(s), len(s)):
                for j in range(-len(s), len(s)):
                    for k in [-2, -1, 1, 2]:
                        sl = slice(i, j, k)
                        self.assertEqual(pyfunc(s, sl), cfunc(s, sl), "'%s'[%d:%d:%d]?" % (s, i, j, k))

    def test_getitem_slice3_kind(self):
        pyfunc = getitem_check_kind_usecase
        cfunc = njit(pyfunc)
        samples = ['abcሴሴ', 'aሴbሴc¡¡¡着着着', '¡着¡着¡着', '着a着b着c', '¡着a¡着b¡着c', '¡着a着¡c']
        for s in samples:
            for i in range(-len(s), len(s)):
                for j in range(-len(s), len(s)):
                    for k in [-2, -1, 1, 2]:
                        sl = slice(i, j, k)
                        self.assertEqual(pyfunc(s, sl), cfunc(s, sl), "'%s'[%d:%d:%d]?" % (s, i, j, k))

    def test_slice3_error(self):
        pyfunc = getitem_usecase
        cfunc = njit(pyfunc)
        for s in UNICODE_EXAMPLES:
            for i in [-2, -1, len(s), len(s) + 1]:
                for j in [-2, -1, len(s), len(s) + 1]:
                    for k in [-2, -1, 1, 2]:
                        sl = slice(i, j, k)
                        self.assertEqual(pyfunc(s, sl), cfunc(s, sl), "'%s'[%d:%d:%d]?" % (s, i, j, k))

    def test_slice_ascii_flag(self):
        """
        Make sure ascii flag is False when ascii and non-ascii characters are
        mixed in output of Unicode slicing.
        """

        @njit
        def f(s):
            return (s[::2]._is_ascii, s[1::2]._is_ascii)
        s = '¿abc¡Y tú, quién te cre\t\tes?'
        self.assertEqual(f(s), (0, 1))

    def test_zfill(self):
        pyfunc = zfill_usecase
        cfunc = njit(pyfunc)
        ZFILL_INPUTS = ['ascii', '+ascii', '-ascii', '-asc ii-', '12345', '-12345', '+12345', '', '¡Y tú crs?', '🐍⚡', '+🐍⚡', '-🐍⚡', '大眼，小手。', '+大眼，小手。', '-大眼，小手。']
        with self.assertRaises(TypingError) as raises:
            cfunc(ZFILL_INPUTS[0], 1.1)
        self.assertIn('<width> must be an Integer', str(raises.exception))
        for s in ZFILL_INPUTS:
            for width in range(-3, 20):
                self.assertEqual(pyfunc(s, width), cfunc(s, width))

    def test_concat(self, flags=no_pyobj_flags):
        pyfunc = concat_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for b in UNICODE_EXAMPLES[::-1]:
                self.assertEqual(pyfunc(a, b), cfunc(a, b), "'%s' + '%s'?" % (a, b))

    def test_repeat(self, flags=no_pyobj_flags):
        pyfunc = repeat_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for b in (-1, 0, 1, 2, 3, 4, 5, 7, 8, 15, 70):
                self.assertEqual(pyfunc(a, b), cfunc(a, b))
                self.assertEqual(pyfunc(b, a), cfunc(b, a))

    def test_repeat_exception_float(self):
        self.disable_leak_check()
        cfunc = njit(repeat_usecase)
        with self.assertRaises(TypingError) as raises:
            cfunc('hi', 2.5)
        self.assertIn(_header_lead + ' Function(<built-in function mul>)', str(raises.exception))

    def test_split_exception_empty_sep(self):
        self.disable_leak_check()
        pyfunc = split_usecase
        cfunc = njit(pyfunc)
        for func in [pyfunc, cfunc]:
            with self.assertRaises(ValueError) as raises:
                func('a', '')
            self.assertIn('empty separator', str(raises.exception))

    def test_split_exception_noninteger_maxsplit(self):
        pyfunc = split_with_maxsplit_usecase
        cfunc = njit(pyfunc)
        for sep in [' ', None]:
            with self.assertRaises(TypingError) as raises:
                cfunc('a', sep, 2.4)
            self.assertIn('float64', str(raises.exception), 'non-integer maxsplit with sep = %s' % sep)

    def test_split(self):
        pyfunc = split_usecase
        cfunc = njit(pyfunc)
        CASES = [(' a ', None), ('', '⚡'), ('abcabc', '⚡'), ('🐍⚡', '⚡'), ('🐍⚡🐍', '⚡'), ('abababa', 'a'), ('abababa', 'b'), ('abababa', 'c'), ('abababa', 'ab'), ('abababa', 'aba')]
        for test_str, splitter in CASES:
            self.assertEqual(pyfunc(test_str, splitter), cfunc(test_str, splitter), "'%s'.split('%s')?" % (test_str, splitter))

    def test_split_with_maxsplit(self):
        CASES = [(' a ', None, 1), ('', '⚡', 1), ('abcabc', '⚡', 1), ('🐍⚡', '⚡', 1), ('🐍⚡🐍', '⚡', 1), ('abababa', 'a', 2), ('abababa', 'b', 1), ('abababa', 'c', 2), ('abababa', 'ab', 1), ('abababa', 'aba', 5)]
        for pyfunc, fmt_str in [(split_with_maxsplit_usecase, "'%s'.split('%s', %d)?"), (split_with_maxsplit_kwarg_usecase, "'%s'.split('%s', maxsplit=%d)?")]:
            cfunc = njit(pyfunc)
            for test_str, splitter, maxsplit in CASES:
                self.assertEqual(pyfunc(test_str, splitter, maxsplit), cfunc(test_str, splitter, maxsplit), fmt_str % (test_str, splitter, maxsplit))

    def test_split_whitespace(self):
        pyfunc = split_whitespace_usecase
        cfunc = njit(pyfunc)
        all_whitespace = ''.join(map(chr, [9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 133, 160, 5760, 8192, 8193, 8194, 8195, 8196, 8197, 8198, 8199, 8200, 8201, 8202, 8232, 8233, 8239, 8287, 12288]))
        CASES = ['', 'abcabc', '🐍 ⚡', '🐍 ⚡ 🐍', '🐍   ⚡ 🐍  ', '  🐍   ⚡ 🐍', ' 🐍' + all_whitespace + '⚡ 🐍  ']
        for test_str in CASES:
            self.assertEqual(pyfunc(test_str), cfunc(test_str), "'%s'.split()?" % (test_str,))

    def test_split_exception_invalid_keepends(self):
        pyfunc = splitlines_with_keepends_usecase
        cfunc = njit(pyfunc)
        accepted_types = (types.Integer, int, types.Boolean, bool)
        for ty, keepends in (('none', None), ('unicode_type', 'None')):
            with self.assertRaises(TypingError) as raises:
                cfunc('\n', keepends)
            msg = '"keepends" must be {}, not {}'.format(accepted_types, ty)
            self.assertIn(msg, str(raises.exception))

    def test_splitlines(self):
        pyfunc = splitlines_usecase
        cfunc = njit(pyfunc)
        cases = ['', '\n', 'abc\r\rabc\r\n', '🐍⚡\x0b', '\x0c🐍⚡\x0c\x0b\x0b🐍\x85', '\u2028aba\u2029baba', '\n\r\na\x0b\x0cb\x0b\x0cc\x1c\x1d\x1e']
        msg = 'Results of "{}".splitlines() must be equal'
        for s in cases:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_splitlines_with_keepends(self):
        pyfuncs = [splitlines_with_keepends_usecase, splitlines_with_keepends_kwarg_usecase]
        messages = ['Results of "{}".splitlines({}) must be equal', 'Results of "{}".splitlines(keepends={}) must be equal']
        cases = ['', '\n', 'abc\r\rabc\r\n', '🐍⚡\x0b', '\x0c🐍⚡\x0c\x0b\x0b🐍\x85', '\u2028aba\u2029baba', '\n\r\na\x0b\x0cb\x0b\x0cc\x1c\x1d\x1e']
        all_keepends = [True, False, 0, 1, -1, 100]
        for pyfunc, msg in zip(pyfuncs, messages):
            cfunc = njit(pyfunc)
            for s, keepends in product(cases, all_keepends):
                self.assertEqual(pyfunc(s, keepends), cfunc(s, keepends), msg=msg.format(s, keepends))

    def test_rsplit_exception_empty_sep(self):
        self.disable_leak_check()
        pyfunc = rsplit_usecase
        cfunc = njit(pyfunc)
        for func in [pyfunc, cfunc]:
            with self.assertRaises(ValueError) as raises:
                func('a', '')
            self.assertIn('empty separator', str(raises.exception))

    def test_rsplit_exception_noninteger_maxsplit(self):
        pyfunc = rsplit_with_maxsplit_usecase
        cfunc = njit(pyfunc)
        accepted_types = (types.Integer, int)
        for sep in [' ', None]:
            with self.assertRaises(TypingError) as raises:
                cfunc('a', sep, 2.4)
            msg = '"maxsplit" must be {}, not float'.format(accepted_types)
            self.assertIn(msg, str(raises.exception))

    def test_rsplit(self):
        pyfunc = rsplit_usecase
        cfunc = njit(pyfunc)
        CASES = [(' a ', None), ('', '⚡'), ('abcabc', '⚡'), ('🐍⚡', '⚡'), ('🐍⚡🐍', '⚡'), ('abababa', 'a'), ('abababa', 'b'), ('abababa', 'c'), ('abababa', 'ab'), ('abababa', 'aba')]
        msg = 'Results of "{}".rsplit("{}") must be equal'
        for s, sep in CASES:
            self.assertEqual(pyfunc(s, sep), cfunc(s, sep), msg=msg.format(s, sep))

    def test_rsplit_with_maxsplit(self):
        pyfuncs = [rsplit_with_maxsplit_usecase, rsplit_with_maxsplit_kwarg_usecase]
        CASES = [(' a ', None, 1), ('', '⚡', 1), ('abcabc', '⚡', 1), ('🐍⚡', '⚡', 1), ('🐍⚡🐍', '⚡', 1), ('abababa', 'a', 2), ('abababa', 'b', 1), ('abababa', 'c', 2), ('abababa', 'ab', 1), ('abababa', 'aba', 5)]
        messages = ['Results of "{}".rsplit("{}", {}) must be equal', 'Results of "{}".rsplit("{}", maxsplit={}) must be equal']
        for pyfunc, msg in zip(pyfuncs, messages):
            cfunc = njit(pyfunc)
            for test_str, sep, maxsplit in CASES:
                self.assertEqual(pyfunc(test_str, sep, maxsplit), cfunc(test_str, sep, maxsplit), msg=msg.format(test_str, sep, maxsplit))

    def test_rsplit_whitespace(self):
        pyfunc = rsplit_whitespace_usecase
        cfunc = njit(pyfunc)
        all_whitespace = ''.join(map(chr, [9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 133, 160, 5760, 8192, 8193, 8194, 8195, 8196, 8197, 8198, 8199, 8200, 8201, 8202, 8232, 8233, 8239, 8287, 12288]))
        CASES = ['', 'abcabc', '🐍 ⚡', '🐍 ⚡ 🐍', '🐍   ⚡ 🐍  ', '  🐍   ⚡ 🐍', ' 🐍' + all_whitespace + '⚡ 🐍  ']
        msg = 'Results of "{}".rsplit() must be equal'
        for s in CASES:
            self.assertEqual(pyfunc(s), cfunc(s), msg.format(s))

    def test_join_empty(self):
        pyfunc = join_empty_usecase
        cfunc = njit(pyfunc)
        CASES = ['', '🐍🐍🐍']
        for sep in CASES:
            self.assertEqual(pyfunc(sep), cfunc(sep), "'%s'.join([])?" % (sep,))

    def test_join_non_string_exception(self):
        pyfunc = join_usecase
        cfunc = njit(pyfunc)
        with self.assertRaises(TypingError) as raises:
            cfunc('', [1, 2, 3])
        exc_message = str(raises.exception)
        self.assertIn('During: resolving callee type: BoundFunction', exc_message)
        self.assertIn('reflected list(int', exc_message)

    def test_join(self):
        pyfunc = join_usecase
        cfunc = njit(pyfunc)
        CASES = [('', ['', '', '']), ('a', ['', '', '']), ('', ['a', 'bbbb', 'c']), ('🐍🐍🐍', ['⚡⚡'] * 5)]
        for sep, parts in CASES:
            self.assertEqual(pyfunc(sep, parts), cfunc(sep, parts), "'%s'.join('%s')?" % (sep, parts))

    def test_join_interleave_str(self):
        pyfunc = join_usecase
        cfunc = njit(pyfunc)
        CASES = [('abc', '123'), ('🐍🐍🐍', '⚡⚡')]
        for sep, parts in CASES:
            self.assertEqual(pyfunc(sep, parts), cfunc(sep, parts), "'%s'.join('%s')?" % (sep, parts))

    def test_justification(self):
        for pyfunc, case_name in [(center_usecase, 'center'), (ljust_usecase, 'ljust'), (rjust_usecase, 'rjust')]:
            cfunc = njit(pyfunc)
            with self.assertRaises(TypingError) as raises:
                cfunc(UNICODE_EXAMPLES[0], 1.1)
            self.assertIn('The width must be an Integer', str(raises.exception))
            for s in UNICODE_EXAMPLES:
                for width in range(-3, 20):
                    self.assertEqual(pyfunc(s, width), cfunc(s, width), "'%s'.%s(%d)?" % (s, case_name, width))

    def test_justification_fillchar(self):
        for pyfunc, case_name in [(center_usecase_fillchar, 'center'), (ljust_usecase_fillchar, 'ljust'), (rjust_usecase_fillchar, 'rjust')]:
            cfunc = njit(pyfunc)
            for fillchar in [' ', '+', 'ú', '处']:
                with self.assertRaises(TypingError) as raises:
                    cfunc(UNICODE_EXAMPLES[0], 1.1, fillchar)
                self.assertIn('The width must be an Integer', str(raises.exception))
                for s in UNICODE_EXAMPLES:
                    for width in range(-3, 20):
                        self.assertEqual(pyfunc(s, width, fillchar), cfunc(s, width, fillchar), "'%s'.%s(%d, '%s')?" % (s, case_name, width, fillchar))

    def test_justification_fillchar_exception(self):
        self.disable_leak_check()
        for pyfunc in [center_usecase_fillchar, ljust_usecase_fillchar, rjust_usecase_fillchar]:
            cfunc = njit(pyfunc)
            for fillchar in ['', '+0', 'quién', '处着']:
                with self.assertRaises(ValueError) as raises:
                    cfunc(UNICODE_EXAMPLES[0], 20, fillchar)
                self.assertIn('The fill character must be exactly one', str(raises.exception))
            for fillchar in [1, 1.1]:
                with self.assertRaises(TypingError) as raises:
                    cfunc(UNICODE_EXAMPLES[0], 20, fillchar)
                self.assertIn('The fillchar must be a UnicodeType', str(raises.exception))

    def test_inplace_concat(self, flags=no_pyobj_flags):
        pyfunc = inplace_concat_usecase
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            for b in UNICODE_EXAMPLES[::-1]:
                self.assertEqual(pyfunc(a, b), cfunc(a, b), "'%s' + '%s'?" % (a, b))

    def test_isidentifier(self):

        def pyfunc(s):
            return s.isidentifier()
        cfunc = njit(pyfunc)
        cpython = ['a', 'Z', '_', 'b0', 'bc', 'b_', 'µ', '𝔘𝔫𝔦𝔠𝔬𝔡𝔢', ' ', '[', '©', '0']
        cpython_extras = ['\ud800', '\udfff', '\ud800\ud800', '\udfff\udfff', 'a\ud800b\udfff', 'a\udfffb\ud800', 'a\ud800b\udfffa', 'a\udfffb\ud800a']
        msg = 'Results of "{}".isidentifier() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_strip(self):
        STRIP_CASES = [('ass cii', 'ai'), ('ass cii', None), ('asscii', 'ai '), ('asscii ', 'ai '), (' asscii  ', 'ai '), (' asscii  ', 'asci '), (' asscii  ', 's'), ('      ', ' '), ('', ' '), ('', ''), ('', None), (' ', None), ('  asscii  ', 'ai '), ('  asscii  ', ''), ('  asscii  ', None), ('tú quién te crees?', 'étú? '), ('  tú quién te crees?   ', 'étú? '), ('  tú qrees?   ', ''), ('  tú quién te crees?   ', None), ('大处 着眼，小处着手。大大大处', '大处'), (' 大处大处  ', ''), ('\t\nabcd\t', '\ta'), (' 大处大处  ', None), ('\t abcd \t', None), ('\n abcd \n', None), ('\r abcd \r', None), ('\x0b abcd \x0b', None), ('\x0c abcd \x0c', None), ('\u2029abcd\u205f', None), ('\x85abcd\u2009', None)]
        for pyfunc, case_name in [(strip_usecase, 'strip'), (lstrip_usecase, 'lstrip'), (rstrip_usecase, 'rstrip')]:
            cfunc = njit(pyfunc)
            for string, chars in STRIP_CASES:
                self.assertEqual(pyfunc(string), cfunc(string), "'%s'.%s()?" % (string, case_name))
        for pyfunc, case_name in [(strip_usecase_chars, 'strip'), (lstrip_usecase_chars, 'lstrip'), (rstrip_usecase_chars, 'rstrip')]:
            cfunc = njit(pyfunc)
            sig1 = types.unicode_type(types.unicode_type, types.Optional(types.unicode_type))
            cfunc_optional = njit([sig1])(pyfunc)

            def try_compile_bad_optional(*args):
                bad = types.unicode_type(types.unicode_type, types.Optional(types.float64))
                njit([bad])(pyfunc)
            for fn in (cfunc, try_compile_bad_optional):
                with self.assertRaises(TypingError) as raises:
                    fn('tú quis?', 1.1)
                self.assertIn('The arg must be a UnicodeType or None', str(raises.exception))
            for fn in (cfunc, cfunc_optional):
                for string, chars in STRIP_CASES:
                    self.assertEqual(pyfunc(string, chars), fn(string, chars), "'%s'.%s('%s')?" % (string, case_name, chars))

    def test_isspace(self):

        def pyfunc(s):
            return s.isspace()
        cfunc = njit(pyfunc)
        cpython = ['\u2000', '\u200a', '—', '𐐁', '𐐧', '𐐩', '𐑎', '🐍', '👯']
        cpython_extras = ['\ud800', '\udfff', '\ud800\ud800', '\udfff\udfff', 'a\ud800b\udfff', 'a\udfffb\ud800', 'a\ud800b\udfffa', 'a\udfffb\ud800a']
        msg = 'Results of "{}".isspace() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_istitle(self):
        pyfunc = istitle_usecase
        cfunc = njit(pyfunc)
        error_msg = "'{0}'.py_istitle() = {1}\n'{0}'.c_istitle() = {2}"
        unicode_title = [x.title() for x in UNICODE_EXAMPLES]
        special = ['', '    ', '  AA  ', '  Ab  ', '1', 'A123', 'A12Bcd', '+abA', '12Abc', 'A12abc', '%^Abc 5 $% Def𐐁𐐩', '𐐧𐑎', '𐐩', '𐑎', '🐍 Is', '🐍 NOT', '👯Is', 'ῼ', 'Greek ῼitlecases ...']
        ISTITLE_EXAMPLES = UNICODE_EXAMPLES + unicode_title + special
        for s in ISTITLE_EXAMPLES:
            py_result = pyfunc(s)
            c_result = cfunc(s)
            self.assertEqual(py_result, c_result, error_msg.format(s, py_result, c_result))

    def test_isprintable(self):

        def pyfunc(s):
            return s.isprintable()
        cfunc = njit(pyfunc)
        cpython = ['', ' ', 'abcdefg', 'abcdefg\n', 'ʹ', '\u0378', '\ud800', '👯', '\U000e0020']
        msg = 'Results of "{}".isprintable() must be equal'
        for s in UNICODE_EXAMPLES + cpython:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_pointless_slice(self, flags=no_pyobj_flags):

        def pyfunc(a):
            return a[:]
        cfunc = njit(pyfunc)
        args = ['a']
        self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_walk_backwards(self, flags=no_pyobj_flags):

        def pyfunc(a):
            return a[::-1]
        cfunc = njit(pyfunc)
        args = ['a']
        self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_stride_slice(self, flags=no_pyobj_flags):

        def pyfunc(a):
            return a[::2]
        cfunc = njit(pyfunc)
        args = ['a']
        self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_basic_lt(self, flags=no_pyobj_flags):

        def pyfunc(a, b):
            return a < b
        cfunc = njit(pyfunc)
        args = ['ab', 'b']
        self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_basic_gt(self, flags=no_pyobj_flags):

        def pyfunc(a, b):
            return a > b
        cfunc = njit(pyfunc)
        args = ['ab', 'b']
        self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_comparison(self):

        def pyfunc(option, x, y):
            if option == '==':
                return x == y
            elif option == '!=':
                return x != y
            elif option == '<':
                return x < y
            elif option == '>':
                return x > y
            elif option == '<=':
                return x <= y
            elif option == '>=':
                return x >= y
            else:
                return None
        cfunc = njit(pyfunc)
        for x, y in permutations(UNICODE_ORDERING_EXAMPLES, r=2):
            for cmpop in ['==', '!=', '<', '>', '<=', '>=', '']:
                args = [cmpop, x, y]
                self.assertEqual(pyfunc(*args), cfunc(*args), msg='failed on {}'.format(args))

    def test_literal_concat(self):

        def pyfunc(x):
            abc = 'abc'
            if len(x):
                return abc + 'b123' + x + 'IO'
            else:
                return x + abc + '123' + x
        cfunc = njit(pyfunc)
        args = ['x']
        self.assertEqual(pyfunc(*args), cfunc(*args))
        args = ['']
        self.assertEqual(pyfunc(*args), cfunc(*args))

    def test_literal_comparison(self):

        def pyfunc(option):
            x = 'a123'
            y = 'aa12'
            if option == '==':
                return x == y
            elif option == '!=':
                return x != y
            elif option == '<':
                return x < y
            elif option == '>':
                return x > y
            elif option == '<=':
                return x <= y
            elif option == '>=':
                return x >= y
            else:
                return None
        cfunc = njit(pyfunc)
        for cmpop in ['==', '!=', '<', '>', '<=', '>=', '']:
            args = [cmpop]
            self.assertEqual(pyfunc(*args), cfunc(*args), msg='failed on {}'.format(args))

    def test_literal_len(self):

        def pyfunc():
            return len('abc')
        cfunc = njit(pyfunc)
        self.assertEqual(pyfunc(), cfunc())

    def test_literal_getitem(self):

        def pyfunc(which):
            return 'abc'[which]
        cfunc = njit(pyfunc)
        for a in [-1, 0, 1, slice(1, None), slice(None, -1)]:
            args = [a]
            self.assertEqual(pyfunc(*args), cfunc(*args), msg='failed on {}'.format(args))

    def test_literal_in(self):

        def pyfunc(x):
            return x in '9876zabiuh'
        cfunc = njit(pyfunc)
        for a in ['a', '9', '1', '', '8uha', '987']:
            args = [a]
            self.assertEqual(pyfunc(*args), cfunc(*args), msg='failed on {}'.format(args))

    def test_literal_xyzwith(self):

        def pyfunc(x, y):
            return ('abc'.startswith(x), 'cde'.endswith(y))
        cfunc = njit(pyfunc)
        for args in permutations('abcdefg', r=2):
            self.assertEqual(pyfunc(*args), cfunc(*args), msg='failed on {}'.format(args))

    def test_literal_find(self):

        def pyfunc(x):
            return ('abc'.find(x), x.find('a'))
        cfunc = njit(pyfunc)
        for a in ['ab']:
            args = [a]
            self.assertEqual(pyfunc(*args), cfunc(*args), msg='failed on {}'.format(args))

    def test_not(self):

        def pyfunc(x):
            return not x
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            args = [a]
            self.assertEqual(pyfunc(*args), cfunc(*args), msg='failed on {}'.format(args))

    def test_capitalize(self):

        def pyfunc(x):
            return x.capitalize()
        cfunc = njit(pyfunc)
        cpython = ['𐑏', '𐑏𐑏', '𐐧𐑏', '𐑏𐐧', 'X𐐧x𐑏', 'hİ', 'ῒİ', 'ﬁnnish', 'AͅΣ']
        cpython_extras = ['𐀀\U00100000']
        msg = 'Results of "{}".capitalize() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_isupper(self):

        def pyfunc(x):
            return x.isupper()
        cfunc = njit(pyfunc)
        uppers = [x.upper() for x in UNICODE_EXAMPLES]
        extras = ['AA12A', 'aa12a', '大AA12A', '大aa12a', 'AAAǄA', 'A 1 1 大']
        cpython = ['Ⅷ', 'ⅷ', '𐐁', '𐐧', '𐐩', '𐑎', '🐍', '👯']
        fourxcpy = [x * 4 for x in cpython]
        for a in UNICODE_EXAMPLES + uppers + extras + cpython + fourxcpy:
            args = [a]
            self.assertEqual(pyfunc(*args), cfunc(*args), msg='failed on {}'.format(args))

    def test_upper(self):

        def pyfunc(x):
            return x.upper()
        cfunc = njit(pyfunc)
        for a in UNICODE_EXAMPLES:
            args = [a]
            self.assertEqual(pyfunc(*args), cfunc(*args), msg='failed on {}'.format(args))

    def test_casefold(self):

        def pyfunc(x):
            return x.casefold()
        cfunc = njit(pyfunc)
        cpython = ['hello', 'hELlo', 'ß', 'ﬁ', 'Σ', 'AͅΣ', 'µ']
        cpython_extras = ['𐀀\U00100000']
        msg = 'Results of "{}".casefold() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_isalpha(self):

        def pyfunc(x):
            return x.isalpha()
        cfunc = njit(pyfunc)
        cpython = ['ῼ', '𐐁', '𐐧', '𐐩', '𐑎', '🐍', '👯']
        extras = ['\ud800', '\udfff', '\ud800\ud800', '\udfff\udfff', 'a\ud800b\udfff', 'a\udfffb\ud800', 'a\ud800b\udfffa', 'a\udfffb\ud800a']
        msg = 'Results of "{}".isalpha() must be equal'
        for s in UNICODE_EXAMPLES + [''] + extras + cpython:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_isascii(self):

        def pyfunc(x):
            return x.isascii()
        cfunc = njit(pyfunc)
        cpython = ['', '\x00', '\x7f', '\x00\x7f', '\x80', 'é', ' ']
        msg = 'Results of "{}".isascii() must be equal'
        for s in UNICODE_EXAMPLES + cpython:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_title(self):
        pyfunc = title
        cfunc = njit(pyfunc)
        cpython = ['𐑏', '𐑏𐑏', '𐑏𐑏 𐑏𐑏', '𐐧𐑏 𐐧𐑏', '𐑏𐐧 𐑏𐐧', 'X𐐧x𐑏 X𐐧x𐑏', 'ﬁNNISH', 'AΣ ᾡxy', 'AΣA']
        msg = 'Results of "{}".title() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_swapcase(self):

        def pyfunc(x):
            return x.swapcase()
        cfunc = njit(pyfunc)
        cpython = ['𐑏', '𐐧', '𐑏𐑏', '𐐧𐑏', '𐑏𐐧', 'X𐐧x𐑏', 'ﬁ', 'İ', 'Σ', 'ͅΣ', 'AͅΣ', 'AͅΣa', 'AͅΣ', 'AΣͅ', 'Σͅ ', 'Σ', 'ß', 'ῒ']
        cpython_extras = ['𐀀\U00100000']
        msg = 'Results of "{}".swapcase() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_islower(self):
        pyfunc = islower_usecase
        cfunc = njit(pyfunc)
        lowers = [x.lower() for x in UNICODE_EXAMPLES]
        extras = ['AA12A', 'aa12a', '大AA12A', '大aa12a', 'AAAǄA', 'A 1 1 大']
        cpython = ['Ⅷ', 'ⅷ', '𐐁', '𐐧', '𐐩', '𐑎', '🐍', '👯']
        cpython += [x * 4 for x in cpython]
        msg = 'Results of "{}".islower() must be equal'
        for s in UNICODE_EXAMPLES + lowers + [''] + extras + cpython:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_isalnum(self):

        def pyfunc(x):
            return x.isalnum()
        cfunc = njit(pyfunc)
        cpython = ['𐐁', '𐐧', '𐐩', '𐑎', '𝟶', '𑁦', '𐒠', '🄇']
        extras = ['\ud800', '\udfff', '\ud800\ud800', '\udfff\udfff', 'a\ud800b\udfff', 'a\udfffb\ud800', 'a\ud800b\udfffa', 'a\udfffb\ud800a']
        msg = 'Results of "{}".isalnum() must be equal'
        for s in UNICODE_EXAMPLES + [''] + extras + cpython:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_lower(self):
        pyfunc = lower_usecase
        cfunc = njit(pyfunc)
        extras = ['AA12A', 'aa12a', '大AA12A', '大aa12a', 'AAAǄA', 'A 1 1 大']
        cpython = ['𐐁', '𐐧', '𐑎', '👯', '𐐧𐐧', '𐐧𐑏', 'X𐐧x𐑏', 'İ']
        sigma = ['Σ', 'ͅΣ', 'AͅΣ', 'AͅΣa', 'Σͅ ', '\U0008fffe', 'ⅷ']
        extra_sigma = 'AΣ\u03a2'
        sigma.append(extra_sigma)
        msg = 'Results of "{}".lower() must be equal'
        for s in UNICODE_EXAMPLES + [''] + extras + cpython + sigma:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_isnumeric(self):

        def pyfunc(x):
            return x.isnumeric()
        cfunc = njit(pyfunc)
        cpython = ['', 'a', '0', '①', '¼', '٠', '0123456789', '0123456789a', '𐐁', '𐐧', '𐐩', '𐑎', '🐍', '👯', '𑁥', '𝟶', '𑁦', '𐒠', '🄇']
        cpython_extras = ['\ud800', '\udfff', '\ud800\ud800', '\udfff\udfff', 'a\ud800b\udfff', 'a\udfffb\ud800', 'a\ud800b\udfffa', 'a\udfffb\ud800a']
        msg = 'Results of "{}".isnumeric() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_isdigit(self):

        def pyfunc(x):
            return x.isdigit()
        cfunc = njit(pyfunc)
        cpython = ['①', '¼', '٠', '𐐁', '𐐧', '𐐩', '𐑎', '🐍', '👯', '𑁥', '𝟶', '𑁦', '𐒠', '🄇']
        cpython_extras = ['\ud800', '\udfff', '\ud800\ud800', '\udfff\udfff', 'a\ud800b\udfff', 'a\udfffb\ud800', 'a\ud800b\udfffa', 'a\udfffb\ud800a']
        msg = 'Results of "{}".isdigit() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_isdecimal(self):

        def pyfunc(x):
            return x.isdecimal()
        cfunc = njit(pyfunc)
        cpython = ['', 'a', '0', '①', '¼', '٠', '0123456789', '0123456789a', '𐐁', '𐐧', '𐐩', '𐑎', '🐍', '👯', '𑁥', '🄇', '𝟶', '𑁦', '𐒠']
        cpython_extras = ['\ud800', '\udfff', '\ud800\ud800', '\udfff\udfff', 'a\ud800b\udfff', 'a\udfffb\ud800', 'a\ud800b\udfffa', 'a\udfffb\ud800a']
        msg = 'Results of "{}".isdecimal() must be equal'
        for s in UNICODE_EXAMPLES + [''] + cpython + cpython_extras:
            self.assertEqual(pyfunc(s), cfunc(s), msg=msg.format(s))

    def test_replace(self):
        pyfunc = replace_usecase
        cfunc = njit(pyfunc)
        CASES = [('abc', '', 'A'), ('', '⚡', 'A'), ('abcabc', '⚡', 'A'), ('🐍⚡', '⚡', 'A'), ('🐍⚡🐍', '⚡', 'A'), ('abababa', 'a', 'A'), ('abababa', 'b', 'A'), ('abababa', 'c', 'A'), ('abababa', 'ab', 'A'), ('abababa', 'aba', 'A')]
        for test_str, old_str, new_str in CASES:
            self.assertEqual(pyfunc(test_str, old_str, new_str), cfunc(test_str, old_str, new_str), "'%s'.replace('%s', '%s')?" % (test_str, old_str, new_str))

    def test_replace_with_count(self):
        pyfunc = replace_with_count_usecase
        cfunc = njit(pyfunc)
        CASES = [('abc', '', 'A'), ('', '⚡', 'A'), ('abcabc', '⚡', 'A'), ('🐍⚡', '⚡', 'A'), ('🐍⚡🐍', '⚡', 'A'), ('abababa', 'a', 'A'), ('abababa', 'b', 'A'), ('abababa', 'c', 'A'), ('abababa', 'ab', 'A'), ('abababa', 'aba', 'A')]
        count_test = [-1, 1, 0, 5]
        for test_str, old_str, new_str in CASES:
            for count in count_test:
                self.assertEqual(pyfunc(test_str, old_str, new_str, count), cfunc(test_str, old_str, new_str, count), "'%s'.replace('%s', '%s', '%s')?" % (test_str, old_str, new_str, count))

    def test_replace_unsupported(self):

        def pyfunc(s, x, y, count):
            return s.replace(x, y, count)
        cfunc = njit(pyfunc)
        with self.assertRaises(TypingError) as raises:
            cfunc('ababababab', 'ba', 'qqq', 3.5)
        msg = 'Unsupported parameters. The parameters must be Integer.'
        self.assertIn(msg, str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc('ababababab', 0, 'qqq', 3)
        msg = 'The object must be a UnicodeType.'
        self.assertIn(msg, str(raises.exception))
        with self.assertRaises(TypingError) as raises:
            cfunc('ababababab', 'ba', 0, 3)
        msg = 'The object must be a UnicodeType.'
        self.assertIn(msg, str(raises.exception))