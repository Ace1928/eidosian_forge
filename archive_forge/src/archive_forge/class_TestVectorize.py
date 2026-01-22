import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
class TestVectorize:

    def test_simple(self):

        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b
        f = vectorize(addsubtract)
        r = f([0, 3, 6, 9], [1, 3, 5, 7])
        assert_array_equal(r, [1, 6, 1, 2])

    def test_scalar(self):

        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b
        f = vectorize(addsubtract)
        r = f([0, 3, 6, 9], 5)
        assert_array_equal(r, [5, 8, 1, 4])

    def test_large(self):
        x = np.linspace(-3, 2, 10000)
        f = vectorize(lambda x: x)
        y = f(x)
        assert_array_equal(y, x)

    def test_ufunc(self):
        f = vectorize(math.cos)
        args = np.array([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi])
        r1 = f(args)
        r2 = np.cos(args)
        assert_array_almost_equal(r1, r2)

    def test_keywords(self):

        def foo(a, b=1):
            return a + b
        f = vectorize(foo)
        args = np.array([1, 2, 3])
        r1 = f(args)
        r2 = np.array([2, 3, 4])
        assert_array_equal(r1, r2)
        r1 = f(args, 2)
        r2 = np.array([3, 4, 5])
        assert_array_equal(r1, r2)

    def test_keywords_with_otypes_order1(self):
        f = vectorize(_foo1, otypes=[float])
        r1 = f(np.arange(3.0), 1.0)
        r2 = f(np.arange(3.0))
        assert_array_equal(r1, r2)

    def test_keywords_with_otypes_order2(self):
        f = vectorize(_foo1, otypes=[float])
        r1 = f(np.arange(3.0))
        r2 = f(np.arange(3.0), 1.0)
        assert_array_equal(r1, r2)

    def test_keywords_with_otypes_order3(self):
        f = vectorize(_foo1, otypes=[float])
        r1 = f(np.arange(3.0))
        r2 = f(np.arange(3.0), y=1.0)
        r3 = f(np.arange(3.0))
        assert_array_equal(r1, r2)
        assert_array_equal(r1, r3)

    def test_keywords_with_otypes_several_kwd_args1(self):
        f = vectorize(_foo2, otypes=[float])
        r1 = f(10.4, z=100)
        r2 = f(10.4, y=-1)
        r3 = f(10.4)
        assert_equal(r1, _foo2(10.4, z=100))
        assert_equal(r2, _foo2(10.4, y=-1))
        assert_equal(r3, _foo2(10.4))

    def test_keywords_with_otypes_several_kwd_args2(self):
        f = vectorize(_foo2, otypes=[float])
        r1 = f(z=100, x=10.4, y=-1)
        r2 = f(1, 2, 3)
        assert_equal(r1, _foo2(z=100, x=10.4, y=-1))
        assert_equal(r2, _foo2(1, 2, 3))

    def test_keywords_no_func_code(self):
        import random
        try:
            vectorize(random.randrange)
        except Exception:
            raise AssertionError()

    def test_keywords2_ticket_2100(self):

        def foo(a, b=1):
            return a + b
        f = vectorize(foo)
        args = np.array([1, 2, 3])
        r1 = f(a=args)
        r2 = np.array([2, 3, 4])
        assert_array_equal(r1, r2)
        r1 = f(b=1, a=args)
        assert_array_equal(r1, r2)
        r1 = f(args, b=2)
        r2 = np.array([3, 4, 5])
        assert_array_equal(r1, r2)

    def test_keywords3_ticket_2100(self):

        def mypolyval(x, p):
            _p = list(p)
            res = _p.pop(0)
            while _p:
                res = res * x + _p.pop(0)
            return res
        vpolyval = np.vectorize(mypolyval, excluded=['p', 1])
        ans = [3, 6]
        assert_array_equal(ans, vpolyval(x=[0, 1], p=[1, 2, 3]))
        assert_array_equal(ans, vpolyval([0, 1], p=[1, 2, 3]))
        assert_array_equal(ans, vpolyval([0, 1], [1, 2, 3]))

    def test_keywords4_ticket_2100(self):

        @vectorize
        def f(**kw):
            res = 1.0
            for _k in kw:
                res *= kw[_k]
            return res
        assert_array_equal(f(a=[1, 2], b=[3, 4]), [3, 8])

    def test_keywords5_ticket_2100(self):

        @vectorize
        def f(*v):
            return np.prod(v)
        assert_array_equal(f([1, 2], [3, 4]), [3, 8])

    def test_coverage1_ticket_2100(self):

        def foo():
            return 1
        f = vectorize(foo)
        assert_array_equal(f(), 1)

    def test_assigning_docstring(self):

        def foo(x):
            """Original documentation"""
            return x
        f = vectorize(foo)
        assert_equal(f.__doc__, foo.__doc__)
        doc = 'Provided documentation'
        f = vectorize(foo, doc=doc)
        assert_equal(f.__doc__, doc)

    def test_UnboundMethod_ticket_1156(self):

        class Foo:
            b = 2

            def bar(self, a):
                return a ** self.b
        assert_array_equal(vectorize(Foo().bar)(np.arange(9)), np.arange(9) ** 2)
        assert_array_equal(vectorize(Foo.bar)(Foo(), np.arange(9)), np.arange(9) ** 2)

    def test_execution_order_ticket_1487(self):
        f1 = vectorize(lambda x: x)
        res1a = f1(np.arange(3))
        res1b = f1(np.arange(0.1, 3))
        f2 = vectorize(lambda x: x)
        res2b = f2(np.arange(0.1, 3))
        res2a = f2(np.arange(3))
        assert_equal(res1a, res2a)
        assert_equal(res1b, res2b)

    def test_string_ticket_1892(self):
        f = np.vectorize(lambda x: x)
        s = '0123456789' * 10
        assert_equal(s, f(s))

    def test_cache(self):
        _calls = [0]

        @vectorize
        def f(x):
            _calls[0] += 1
            return x ** 2
        f.cache = True
        x = np.arange(5)
        assert_array_equal(f(x), x * x)
        assert_equal(_calls[0], len(x))

    def test_otypes(self):
        f = np.vectorize(lambda x: x)
        f.otypes = 'i'
        x = np.arange(5)
        assert_array_equal(f(x), x)

    def test_parse_gufunc_signature(self):
        assert_equal(nfb._parse_gufunc_signature('(x)->()'), ([('x',)], [()]))
        assert_equal(nfb._parse_gufunc_signature('(x,y)->()'), ([('x', 'y')], [()]))
        assert_equal(nfb._parse_gufunc_signature('(x),(y)->()'), ([('x',), ('y',)], [()]))
        assert_equal(nfb._parse_gufunc_signature('(x)->(y)'), ([('x',)], [('y',)]))
        assert_equal(nfb._parse_gufunc_signature('(x)->(y),()'), ([('x',)], [('y',), ()]))
        assert_equal(nfb._parse_gufunc_signature('(),(a,b,c),(d)->(d,e)'), ([(), ('a', 'b', 'c'), ('d',)], [('d', 'e')]))
        assert_equal(nfb._parse_gufunc_signature('(x )->()'), ([('x',)], [()]))
        assert_equal(nfb._parse_gufunc_signature('( x , y )->(  )'), ([('x', 'y')], [()]))
        assert_equal(nfb._parse_gufunc_signature('(x),( y) ->()'), ([('x',), ('y',)], [()]))
        assert_equal(nfb._parse_gufunc_signature('(  x)-> (y )  '), ([('x',)], [('y',)]))
        assert_equal(nfb._parse_gufunc_signature(' (x)->( y),( )'), ([('x',)], [('y',), ()]))
        assert_equal(nfb._parse_gufunc_signature('(  ), ( a,  b,c )  ,(  d)   ->   (d  ,  e)'), ([(), ('a', 'b', 'c'), ('d',)], [('d', 'e')]))
        with assert_raises(ValueError):
            nfb._parse_gufunc_signature('(x)(y)->()')
        with assert_raises(ValueError):
            nfb._parse_gufunc_signature('(x),(y)->')
        with assert_raises(ValueError):
            nfb._parse_gufunc_signature('((x))->(x)')

    def test_signature_simple(self):

        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b
        f = vectorize(addsubtract, signature='(),()->()')
        r = f([0, 3, 6, 9], [1, 3, 5, 7])
        assert_array_equal(r, [1, 6, 1, 2])

    def test_signature_mean_last(self):

        def mean(a):
            return a.mean()
        f = vectorize(mean, signature='(n)->()')
        r = f([[1, 3], [2, 4]])
        assert_array_equal(r, [2, 3])

    def test_signature_center(self):

        def center(a):
            return a - a.mean()
        f = vectorize(center, signature='(n)->(n)')
        r = f([[1, 3], [2, 4]])
        assert_array_equal(r, [[-1, 1], [-1, 1]])

    def test_signature_two_outputs(self):
        f = vectorize(lambda x: (x, x), signature='()->(),()')
        r = f([1, 2, 3])
        assert_(isinstance(r, tuple) and len(r) == 2)
        assert_array_equal(r[0], [1, 2, 3])
        assert_array_equal(r[1], [1, 2, 3])

    def test_signature_outer(self):
        f = vectorize(np.outer, signature='(a),(b)->(a,b)')
        r = f([1, 2], [1, 2, 3])
        assert_array_equal(r, [[1, 2, 3], [2, 4, 6]])
        r = f([[[1, 2]]], [1, 2, 3])
        assert_array_equal(r, [[[[1, 2, 3], [2, 4, 6]]]])
        r = f([[1, 0], [2, 0]], [1, 2, 3])
        assert_array_equal(r, [[[1, 2, 3], [0, 0, 0]], [[2, 4, 6], [0, 0, 0]]])
        r = f([1, 2], [[1, 2, 3], [0, 0, 0]])
        assert_array_equal(r, [[[1, 2, 3], [2, 4, 6]], [[0, 0, 0], [0, 0, 0]]])

    def test_signature_computed_size(self):
        f = vectorize(lambda x: x[:-1], signature='(n)->(m)')
        r = f([1, 2, 3])
        assert_array_equal(r, [1, 2])
        r = f([[1, 2, 3], [2, 3, 4]])
        assert_array_equal(r, [[1, 2], [2, 3]])

    def test_signature_excluded(self):

        def foo(a, b=1):
            return a + b
        f = vectorize(foo, signature='()->()', excluded={'b'})
        assert_array_equal(f([1, 2, 3]), [2, 3, 4])
        assert_array_equal(f([1, 2, 3], b=0), [1, 2, 3])

    def test_signature_otypes(self):
        f = vectorize(lambda x: x, signature='(n)->(n)', otypes=['float64'])
        r = f([1, 2, 3])
        assert_equal(r.dtype, np.dtype('float64'))
        assert_array_equal(r, [1, 2, 3])

    def test_signature_invalid_inputs(self):
        f = vectorize(operator.add, signature='(n),(n)->(n)')
        with assert_raises_regex(TypeError, 'wrong number of positional'):
            f([1, 2])
        with assert_raises_regex(ValueError, 'does not have enough dimensions'):
            f(1, 2)
        with assert_raises_regex(ValueError, 'inconsistent size for core dimension'):
            f([1, 2], [1, 2, 3])
        f = vectorize(operator.add, signature='()->()')
        with assert_raises_regex(TypeError, 'wrong number of positional'):
            f(1, 2)

    def test_signature_invalid_outputs(self):
        f = vectorize(lambda x: x[:-1], signature='(n)->(n)')
        with assert_raises_regex(ValueError, 'inconsistent size for core dimension'):
            f([1, 2, 3])
        f = vectorize(lambda x: x, signature='()->(),()')
        with assert_raises_regex(ValueError, 'wrong number of outputs'):
            f(1)
        f = vectorize(lambda x: (x, x), signature='()->()')
        with assert_raises_regex(ValueError, 'wrong number of outputs'):
            f([1, 2])

    def test_size_zero_output(self):
        f = np.vectorize(lambda x: x)
        x = np.zeros([0, 5], dtype=int)
        with assert_raises_regex(ValueError, 'otypes'):
            f(x)
        f.otypes = 'i'
        assert_array_equal(f(x), x)
        f = np.vectorize(lambda x: x, signature='()->()')
        with assert_raises_regex(ValueError, 'otypes'):
            f(x)
        f = np.vectorize(lambda x: x, signature='()->()', otypes='i')
        assert_array_equal(f(x), x)
        f = np.vectorize(lambda x: x, signature='(n)->(n)', otypes='i')
        assert_array_equal(f(x), x)
        f = np.vectorize(lambda x: x, signature='(n)->(n)')
        assert_array_equal(f(x.T), x.T)
        f = np.vectorize(lambda x: [x], signature='()->(n)', otypes='i')
        with assert_raises_regex(ValueError, 'new output dimensions'):
            f(x)

    def test_subclasses(self):

        class subclass(np.ndarray):
            pass
        m = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).view(subclass)
        v = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).view(subclass)
        matvec = np.vectorize(np.matmul, signature='(m,m),(m)->(m)')
        r = matvec(m, v)
        assert_equal(type(r), subclass)
        assert_equal(r, [[1.0, 3.0, 2.0], [4.0, 6.0, 5.0], [7.0, 9.0, 8.0]])
        mult = np.vectorize(lambda x, y: x * y)
        r = mult(m, v)
        assert_equal(type(r), subclass)
        assert_equal(r, m * v)

    def test_name(self):

        @np.vectorize
        def f2(a, b):
            return a + b
        assert f2.__name__ == 'f2'

    def test_decorator(self):

        @vectorize
        def addsubtract(a, b):
            if a > b:
                return a - b
            else:
                return a + b
        r = addsubtract([0, 3, 6, 9], [1, 3, 5, 7])
        assert_array_equal(r, [1, 6, 1, 2])

    def test_docstring(self):

        @vectorize
        def f(x):
            """Docstring"""
            return x
        if sys.flags.optimize < 2:
            assert f.__doc__ == 'Docstring'

    def test_partial(self):

        def foo(x, y):
            return x + y
        bar = partial(foo, 3)
        vbar = np.vectorize(bar)
        assert vbar(1) == 4

    def test_signature_otypes_decorator(self):

        @vectorize(signature='(n)->(n)', otypes=['float64'])
        def f(x):
            return x
        r = f([1, 2, 3])
        assert_equal(r.dtype, np.dtype('float64'))
        assert_array_equal(r, [1, 2, 3])
        assert f.__name__ == 'f'

    def test_bad_input(self):
        with assert_raises(TypeError):
            A = np.vectorize(pyfunc=3)

    def test_no_keywords(self):
        with assert_raises(TypeError):

            @np.vectorize('string')
            def foo():
                return 'bar'

    def test_positional_regression_9477(self):
        f = vectorize(lambda x: x, ['float64'])
        r = f([2])
        assert_equal(r.dtype, np.dtype('float64'))