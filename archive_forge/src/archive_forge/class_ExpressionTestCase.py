import doctest
import os
import pickle
import sys
from tempfile import mkstemp
import unittest
from genshi.core import Markup
from genshi.template.base import Context
from genshi.template.eval import Expression, Suite, Undefined, UndefinedError, \
from genshi.compat import BytesIO, IS_PYTHON2, wrapped_bytes
class ExpressionTestCase(unittest.TestCase):

    def test_eq(self):
        expr = Expression('x,y')
        self.assertEqual(expr, Expression('x,y'))
        self.assertNotEqual(expr, Expression('y, x'))

    def test_hash(self):
        expr = Expression('x,y')
        self.assertEqual(hash(expr), hash(Expression('x,y')))
        self.assertNotEqual(hash(expr), hash(Expression('y, x')))

    def test_pickle(self):
        expr = Expression('1 < 2')
        buf = BytesIO()
        pickle.dump(expr, buf, 2)
        buf.seek(0)
        unpickled = pickle.load(buf)
        assert unpickled.evaluate({}) is True
        assert unpickled.code == expr.code

    def test_name_lookup(self):
        self.assertEqual('bar', Expression('foo').evaluate({'foo': 'bar'}))
        self.assertEqual(id, Expression('id').evaluate({}))
        self.assertEqual('bar', Expression('id').evaluate({'id': 'bar'}))
        self.assertEqual(None, Expression('id').evaluate({'id': None}))

    def test_builtins(self):
        expr = Expression('Markup')
        self.assertEqual(expr.evaluate({}), Markup)

    def test_str_literal(self):
        self.assertEqual('foo', Expression('"foo"').evaluate({}))
        self.assertEqual('foo', Expression('"""foo"""').evaluate({}))
        self.assertEqual(u'foo'.encode('utf-8'), Expression(wrapped_bytes("b'foo'")).evaluate({}))
        self.assertEqual('foo', Expression("'''foo'''").evaluate({}))
        self.assertEqual('foo', Expression("u'foo'").evaluate({}))
        self.assertEqual('foo', Expression("r'foo'").evaluate({}))

    def test_str_literal_non_ascii(self):
        expr = Expression(u"u'þ'")
        self.assertEqual(u'þ', expr.evaluate({}))
        expr = Expression("u'þ'")
        self.assertEqual(u'þ', expr.evaluate({}))
        expr = Expression(wrapped_bytes("b'\\xc3\\xbe'"))
        if IS_PYTHON2:
            self.assertEqual(u'þ', expr.evaluate({}))
        else:
            self.assertEqual(u'þ'.encode('utf-8'), expr.evaluate({}))

    def test_num_literal(self):
        self.assertEqual(42, Expression('42').evaluate({}))
        if IS_PYTHON2:
            self.assertEqual(42, Expression('42L').evaluate({}))
        self.assertEqual(0.42, Expression('.42').evaluate({}))
        if IS_PYTHON2:
            self.assertEqual(7, Expression('07').evaluate({}))
        self.assertEqual(242, Expression('0xF2').evaluate({}))
        self.assertEqual(242, Expression('0XF2').evaluate({}))

    def test_dict_literal(self):
        self.assertEqual({}, Expression('{}').evaluate({}))
        self.assertEqual({'key': True}, Expression("{'key': value}").evaluate({'value': True}))

    def test_list_literal(self):
        self.assertEqual([], Expression('[]').evaluate({}))
        self.assertEqual([1, 2, 3], Expression('[1, 2, 3]').evaluate({}))
        self.assertEqual([True], Expression('[value]').evaluate({'value': True}))

    def test_tuple_literal(self):
        self.assertEqual((), Expression('()').evaluate({}))
        self.assertEqual((1, 2, 3), Expression('(1, 2, 3)').evaluate({}))
        self.assertEqual((True,), Expression('(value,)').evaluate({'value': True}))

    def test_unaryop_pos(self):
        self.assertEqual(1, Expression('+1').evaluate({}))
        self.assertEqual(1, Expression('+x').evaluate({'x': 1}))

    def test_unaryop_neg(self):
        self.assertEqual(-1, Expression('-1').evaluate({}))
        self.assertEqual(-1, Expression('-x').evaluate({'x': 1}))

    def test_unaryop_not(self):
        self.assertEqual(False, Expression('not True').evaluate({}))
        self.assertEqual(False, Expression('not x').evaluate({'x': True}))

    def test_unaryop_inv(self):
        self.assertEqual(-2, Expression('~1').evaluate({}))
        self.assertEqual(-2, Expression('~x').evaluate({'x': 1}))

    def test_binop_add(self):
        self.assertEqual(3, Expression('2 + 1').evaluate({}))
        self.assertEqual(3, Expression('x + y').evaluate({'x': 2, 'y': 1}))

    def test_binop_sub(self):
        self.assertEqual(1, Expression('2 - 1').evaluate({}))
        self.assertEqual(1, Expression('x - y').evaluate({'x': 1, 'y': 1}))

    def test_binop_sub(self):
        self.assertEqual(1, Expression('2 - 1').evaluate({}))
        self.assertEqual(1, Expression('x - y').evaluate({'x': 2, 'y': 1}))

    def test_binop_mul(self):
        self.assertEqual(4, Expression('2 * 2').evaluate({}))
        self.assertEqual(4, Expression('x * y').evaluate({'x': 2, 'y': 2}))

    def test_binop_pow(self):
        self.assertEqual(4, Expression('2 ** 2').evaluate({}))
        self.assertEqual(4, Expression('x ** y').evaluate({'x': 2, 'y': 2}))

    def test_binop_div(self):
        self.assertEqual(2, Expression('4 / 2').evaluate({}))
        self.assertEqual(2, Expression('x / y').evaluate({'x': 4, 'y': 2}))

    def test_binop_floordiv(self):
        self.assertEqual(1, Expression('3 // 2').evaluate({}))
        self.assertEqual(1, Expression('x // y').evaluate({'x': 3, 'y': 2}))

    def test_binop_mod(self):
        self.assertEqual(1, Expression('3 % 2').evaluate({}))
        self.assertEqual(1, Expression('x % y').evaluate({'x': 3, 'y': 2}))

    def test_binop_and(self):
        self.assertEqual(0, Expression('1 & 0').evaluate({}))
        self.assertEqual(0, Expression('x & y').evaluate({'x': 1, 'y': 0}))

    def test_binop_or(self):
        self.assertEqual(1, Expression('1 | 0').evaluate({}))
        self.assertEqual(1, Expression('x | y').evaluate({'x': 1, 'y': 0}))

    def test_binop_xor(self):
        self.assertEqual(1, Expression('1 ^ 0').evaluate({}))
        self.assertEqual(1, Expression('x ^ y').evaluate({'x': 1, 'y': 0}))

    def test_binop_contains(self):
        self.assertEqual(True, Expression('1 in (1, 2, 3)').evaluate({}))
        self.assertEqual(True, Expression('x in y').evaluate({'x': 1, 'y': (1, 2, 3)}))

    def test_binop_not_contains(self):
        self.assertEqual(True, Expression('4 not in (1, 2, 3)').evaluate({}))
        self.assertEqual(True, Expression('x not in y').evaluate({'x': 4, 'y': (1, 2, 3)}))

    def test_binop_is(self):
        self.assertEqual(True, Expression('1 is 1').evaluate({}))
        self.assertEqual(True, Expression('x is y').evaluate({'x': 1, 'y': 1}))
        self.assertEqual(False, Expression('1 is 2').evaluate({}))
        self.assertEqual(False, Expression('x is y').evaluate({'x': 1, 'y': 2}))

    def test_binop_is_not(self):
        self.assertEqual(True, Expression('1 is not 2').evaluate({}))
        self.assertEqual(True, Expression('x is not y').evaluate({'x': 1, 'y': 2}))
        self.assertEqual(False, Expression('1 is not 1').evaluate({}))
        self.assertEqual(False, Expression('x is not y').evaluate({'x': 1, 'y': 1}))

    def test_boolop_and(self):
        self.assertEqual(False, Expression('True and False').evaluate({}))
        self.assertEqual(False, Expression('x and y').evaluate({'x': True, 'y': False}))

    def test_boolop_or(self):
        self.assertEqual(True, Expression('True or False').evaluate({}))
        self.assertEqual(True, Expression('x or y').evaluate({'x': True, 'y': False}))

    def test_compare_eq(self):
        self.assertEqual(True, Expression('1 == 1').evaluate({}))
        self.assertEqual(True, Expression('x == y').evaluate({'x': 1, 'y': 1}))

    def test_compare_ne(self):
        self.assertEqual(False, Expression('1 != 1').evaluate({}))
        self.assertEqual(False, Expression('x != y').evaluate({'x': 1, 'y': 1}))
        if sys.version < '3':
            self.assertEqual(False, Expression('1 <> 1').evaluate({}))
            self.assertEqual(False, Expression('x <> y').evaluate({'x': 1, 'y': 1}))

    def test_compare_lt(self):
        self.assertEqual(True, Expression('1 < 2').evaluate({}))
        self.assertEqual(True, Expression('x < y').evaluate({'x': 1, 'y': 2}))

    def test_compare_le(self):
        self.assertEqual(True, Expression('1 <= 1').evaluate({}))
        self.assertEqual(True, Expression('x <= y').evaluate({'x': 1, 'y': 1}))

    def test_compare_gt(self):
        self.assertEqual(True, Expression('2 > 1').evaluate({}))
        self.assertEqual(True, Expression('x > y').evaluate({'x': 2, 'y': 1}))

    def test_compare_ge(self):
        self.assertEqual(True, Expression('1 >= 1').evaluate({}))
        self.assertEqual(True, Expression('x >= y').evaluate({'x': 1, 'y': 1}))

    def test_compare_multi(self):
        self.assertEqual(True, Expression('1 != 3 == 3').evaluate({}))
        self.assertEqual(True, Expression('x != y == y').evaluate({'x': 1, 'y': 3}))

    def test_call_function(self):
        self.assertEqual(42, Expression('foo()').evaluate({'foo': lambda: 42}))
        data = {'foo': 'bar'}
        self.assertEqual('BAR', Expression('foo.upper()').evaluate(data))
        data = {'foo': {'bar': range(42)}}
        self.assertEqual(42, Expression('len(foo.bar)').evaluate(data))

    def test_call_keywords(self):
        self.assertEqual(42, Expression('foo(x=bar)').evaluate({'foo': lambda x: x, 'bar': 42}))

    def test_call_star_args(self):
        self.assertEqual(42, Expression('foo(*bar)').evaluate({'foo': lambda x: x, 'bar': [42]}))

    def test_call_dstar_args(self):

        def foo(x):
            return x
        expr = Expression('foo(**bar)')
        self.assertEqual(42, expr.evaluate({'foo': foo, 'bar': {'x': 42}}))

    def test_lambda(self):
        data = {'items': range(5)}
        expr = Expression('filter(lambda x: x > 2, items)')
        self.assertEqual([3, 4], list(expr.evaluate(data)))

    def test_lambda_tuple_arg(self):
        if not IS_PYTHON2:
            return
        data = {'items': [(1, 2), (2, 1)]}
        expr = Expression('filter(lambda (x, y): x > y, items)')
        self.assertEqual([(2, 1)], list(expr.evaluate(data)))

    def test_list_comprehension(self):
        expr = Expression('[n for n in numbers if n < 2]')
        self.assertEqual([0, 1], expr.evaluate({'numbers': range(5)}))
        expr = Expression('[(i, n + 1) for i, n in enumerate(numbers)]')
        self.assertEqual([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], expr.evaluate({'numbers': range(5)}))
        expr = Expression('[offset + n for n in numbers]')
        self.assertEqual([2, 3, 4, 5, 6], expr.evaluate({'numbers': range(5), 'offset': 2}))
        expr = Expression('[n for group in groups for n in group]')
        self.assertEqual([0, 1, 0, 1, 2], expr.evaluate({'groups': [range(2), range(3)]}))
        expr = Expression('[(a, b) for a in x for b in y]')
        self.assertEqual([('x0', 'y0'), ('x0', 'y1'), ('x1', 'y0'), ('x1', 'y1')], expr.evaluate({'x': ['x0', 'x1'], 'y': ['y0', 'y1']}))

    def test_list_comprehension_with_getattr(self):
        items = [{'name': 'a', 'value': 1}, {'name': 'b', 'value': 2}]
        expr = Expression('[i.name for i in items if i.value > 1]')
        self.assertEqual(['b'], expr.evaluate({'items': items}))

    def test_list_comprehension_with_getitem(self):
        items = [{'name': 'a', 'value': 1}, {'name': 'b', 'value': 2}]
        expr = Expression("[i['name'] for i in items if i['value'] > 1]")
        self.assertEqual(['b'], expr.evaluate({'items': items}))

    def test_generator_expression(self):
        expr = Expression('list(n for n in numbers if n < 2)')
        self.assertEqual([0, 1], expr.evaluate({'numbers': range(5)}))
        expr = Expression('list((i, n + 1) for i, n in enumerate(numbers))')
        self.assertEqual([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], expr.evaluate({'numbers': range(5)}))
        expr = Expression('list(offset + n for n in numbers)')
        self.assertEqual([2, 3, 4, 5, 6], expr.evaluate({'numbers': range(5), 'offset': 2}))
        expr = Expression('list(n for group in groups for n in group)')
        self.assertEqual([0, 1, 0, 1, 2], expr.evaluate({'groups': [range(2), range(3)]}))
        expr = Expression('list((a, b) for a in x for b in y)')
        self.assertEqual([('x0', 'y0'), ('x0', 'y1'), ('x1', 'y0'), ('x1', 'y1')], expr.evaluate({'x': ['x0', 'x1'], 'y': ['y0', 'y1']}))

    def test_generator_expression_with_getattr(self):
        items = [{'name': 'a', 'value': 1}, {'name': 'b', 'value': 2}]
        expr = Expression('list(i.name for i in items if i.value > 1)')
        self.assertEqual(['b'], expr.evaluate({'items': items}))

    def test_generator_expression_with_getitem(self):
        items = [{'name': 'a', 'value': 1}, {'name': 'b', 'value': 2}]
        expr = Expression("list(i['name'] for i in items if i['value'] > 1)")
        self.assertEqual(['b'], expr.evaluate({'items': items}))
    if sys.version_info >= (2, 5):

        def test_conditional_expression(self):
            expr = Expression("'T' if foo else 'F'")
            self.assertEqual('T', expr.evaluate({'foo': True}))
            self.assertEqual('F', expr.evaluate({'foo': False}))

    def test_slice(self):
        expr = Expression('numbers[0:2]')
        self.assertEqual([0, 1], expr.evaluate({'numbers': list(range(5))}))

    def test_slice_with_vars(self):
        expr = Expression('numbers[start:end]')
        res = expr.evaluate({'numbers': list(range(5)), 'start': 0, 'end': 2})
        self.assertEqual([0, 1], res)

    def test_slice_copy(self):
        expr = Expression('numbers[:]')
        res = expr.evaluate({'numbers': list(range(5))})
        self.assertEqual([0, 1, 2, 3, 4], res)

    def test_slice_stride(self):
        expr = Expression('numbers[::stride]')
        res = expr.evaluate({'numbers': list(range(5)), 'stride': 2})
        self.assertEqual([0, 2, 4], res)

    def test_slice_negative_start(self):
        expr = Expression('numbers[-1:]')
        self.assertEqual([4], expr.evaluate({'numbers': list(range(5))}))

    def test_slice_negative_end(self):
        expr = Expression('numbers[:-1]')
        res = expr.evaluate({'numbers': list(range(5))})
        self.assertEqual([0, 1, 2, 3], res)

    def test_slice_constant(self):
        expr = Expression('numbers[1]')
        res = expr.evaluate({'numbers': list(range(5))})
        self.assertEqual(res, 1)

    def test_slice_call(self):

        def f():
            return 2
        expr = Expression('numbers[f()]')
        res = expr.evaluate({'numbers': list(range(5)), 'f': f})
        self.assertEqual(res, 2)

    def test_slice_name(self):
        expr = Expression('numbers[v]')
        res = expr.evaluate({'numbers': list(range(5)), 'v': 2})
        self.assertEqual(res, 2)

    def test_slice_attribute(self):

        class ValueHolder:

            def __init__(self):
                self.value = 3
        expr = Expression('numbers[obj.value]')
        res = expr.evaluate({'numbers': list(range(5)), 'obj': ValueHolder()})
        self.assertEqual(res, 3)

    def test_access_undefined(self):
        expr = Expression('nothing', filename='index.html', lineno=50, lookup='lenient')
        retval = expr.evaluate({})
        assert isinstance(retval, Undefined)
        self.assertEqual('nothing', retval._name)
        assert retval._owner is UNDEFINED

    def test_getattr_undefined(self):

        class Something(object):

            def __repr__(self):
                return '<Something>'
        something = Something()
        expr = Expression('something.nil', filename='index.html', lineno=50, lookup='lenient')
        retval = expr.evaluate({'something': something})
        assert isinstance(retval, Undefined)
        self.assertEqual('nil', retval._name)
        assert retval._owner is something

    def test_getattr_exception(self):

        class Something(object):

            def prop_a(self):
                raise NotImplementedError
            prop_a = property(prop_a)

            def prop_b(self):
                raise AttributeError
            prop_b = property(prop_b)
        self.assertRaises(NotImplementedError, Expression('s.prop_a').evaluate, {'s': Something()})
        self.assertRaises(AttributeError, Expression('s.prop_b').evaluate, {'s': Something()})

    def test_getitem_undefined_string(self):

        class Something(object):

            def __repr__(self):
                return '<Something>'
        something = Something()
        expr = Expression('something["nil"]', filename='index.html', lineno=50, lookup='lenient')
        retval = expr.evaluate({'something': something})
        assert isinstance(retval, Undefined)
        self.assertEqual('nil', retval._name)
        assert retval._owner is something

    def test_getitem_exception(self):

        class Something(object):

            def __getitem__(self, key):
                raise NotImplementedError
        self.assertRaises(NotImplementedError, Expression('s["foo"]').evaluate, {'s': Something()})

    def test_error_access_undefined(self):
        expr = Expression('nothing', filename='index.html', lineno=50, lookup='strict')
        try:
            expr.evaluate({})
            self.fail('Expected UndefinedError')
        except UndefinedError as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            frame = exc_traceback.tb_next
            frames = []
            while frame.tb_next:
                frame = frame.tb_next
                frames.append(frame)
            self.assertEqual('"nothing" not defined', str(e))
            self.assertEqual("<Expression 'nothing'>", frames[-3].tb_frame.f_code.co_name)
            self.assertEqual('index.html', frames[-3].tb_frame.f_code.co_filename)
            self.assertEqual(50, frames[-3].tb_lineno)

    def test_error_getattr_undefined(self):

        class Something(object):

            def __repr__(self):
                return '<Something>'
        expr = Expression('something.nil', filename='index.html', lineno=50, lookup='strict')
        try:
            expr.evaluate({'something': Something()})
            self.fail('Expected UndefinedError')
        except UndefinedError as e:
            self.assertEqual('<Something> has no member named "nil"', str(e))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            search_string = "<Expression 'something.nil'>"
            frame = exc_traceback.tb_next
            while frame.tb_next:
                frame = frame.tb_next
                code = frame.tb_frame.f_code
                if code.co_name == search_string:
                    break
            else:
                self.fail('never found the frame I was looking for')
            self.assertEqual('index.html', code.co_filename)
            self.assertEqual(50, frame.tb_lineno)

    def test_error_getitem_undefined_string(self):

        class Something(object):

            def __repr__(self):
                return '<Something>'
        expr = Expression('something["nil"]', filename='index.html', lineno=50, lookup='strict')
        try:
            expr.evaluate({'something': Something()})
            self.fail('Expected UndefinedError')
        except UndefinedError as e:
            self.assertEqual('<Something> has no member named "nil"', str(e))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            search_string = '<Expression \'something["nil"]\'>'
            frame = exc_traceback.tb_next
            while frame.tb_next:
                frame = frame.tb_next
                code = frame.tb_frame.f_code
                if code.co_name == search_string:
                    break
            else:
                self.fail('never found the frame I was looking for')
            self.assertEqual('index.html', code.co_filename)
            self.assertEqual(50, frame.tb_lineno)

    def test_getitem_with_constant_string(self):
        data = dict(dict={'some': 'thing'})
        self.assertEqual('thing', Expression("dict['some']").evaluate(data))

    def test_getitem_with_simple_index(self):
        data = dict(values={True: 'bar', 2.5: 'baz', None: 'quox', 42: 'quooox', b'foo': 'foobar'})
        self.assertEqual('bar', Expression('values[True]').evaluate(data))
        self.assertEqual('baz', Expression('values[2.5]').evaluate(data))
        self.assertEqual('quooox', Expression('values[42]').evaluate(data))
        self.assertEqual('foobar', Expression('values[b"foo"]').evaluate(data))
        self.assertEqual('quox', Expression('values[None]').evaluate(data))

    def test_array_indices(self):
        data = dict(items=[1, 2, 3])
        self.assertEqual(1, Expression('items[0]').evaluate(data))
        self.assertEqual(3, Expression('items[-1]').evaluate(data))

    def test_item_access_for_attributes(self):

        class MyClass(object):
            myattr = 'Bar'
        data = {'mine': MyClass(), 'key': 'myattr'}
        self.assertEqual('Bar', Expression('mine.myattr').evaluate(data))
        self.assertEqual('Bar', Expression('mine["myattr"]').evaluate(data))
        self.assertEqual('Bar', Expression('mine[key]').evaluate(data))

    def test_function_in_item_access(self):
        data = dict(values={'foo': 'bar'})
        self.assertEqual('bar', Expression('values[str("foo")]').evaluate(data))