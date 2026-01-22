import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
class TestArgmap:

    class ArgmapError(RuntimeError):
        pass

    def test_trivial_function(self):

        def do_not_call(x):
            raise ArgmapError('do not call this function')

        @argmap(do_not_call)
        def trivial_argmap():
            return 1
        assert trivial_argmap() == 1

    def test_trivial_iterator(self):

        def do_not_call(x):
            raise ArgmapError('do not call this function')

        @argmap(do_not_call)
        def trivial_argmap():
            yield from (1, 2, 3)
        assert tuple(trivial_argmap()) == (1, 2, 3)

    def test_contextmanager(self):
        container = []

        def contextmanager(x):
            nonlocal container
            return (x, lambda: container.append(x))

        @argmap(contextmanager, 0, 1, 2, try_finally=True)
        def foo(x, y, z):
            return (x, y, z)
        x, y, z = foo('a', 'b', 'c')
        assert container == ['c', 'b', 'a']

    def test_tryfinally_generator(self):
        container = []

        def singleton(x):
            return (x,)
        with pytest.raises(nx.NetworkXError):

            @argmap(singleton, 0, 1, 2, try_finally=True)
            def foo(x, y, z):
                yield from (x, y, z)

        @argmap(singleton, 0, 1, 2)
        def foo(x, y, z):
            return x + y + z
        q = foo('a', 'b', 'c')
        assert q == ('a', 'b', 'c')

    def test_actual_vararg(self):

        @argmap(lambda x: -x, 4)
        def foo(x, y, *args):
            return (x, y) + tuple(args)
        assert foo(1, 2, 3, 4, 5, 6) == (1, 2, 3, 4, -5, 6)

    def test_signature_destroying_intermediate_decorator(self):

        def add_one_to_first_bad_decorator(f):
            """Bad because it doesn't wrap the f signature (clobbers it)"""

            def decorated(a, *args, **kwargs):
                return f(a + 1, *args, **kwargs)
            return decorated
        add_two_to_second = argmap(lambda b: b + 2, 1)

        @add_two_to_second
        @add_one_to_first_bad_decorator
        def add_one_and_two(a, b):
            return (a, b)
        assert add_one_and_two(5, 5) == (6, 7)

    def test_actual_kwarg(self):

        @argmap(lambda x: -x, 'arg')
        def foo(*, arg):
            return arg
        assert foo(arg=3) == -3

    def test_nested_tuple(self):

        def xform(x, y):
            u, v = y
            return (x + u + v, (x + u, x + v))

        @argmap(xform, (0, ('t', 2)))
        def foo(a, *args, **kwargs):
            return (a, args, kwargs)
        a, args, kwargs = foo(1, 2, 3, t=4)
        assert a == 1 + 4 + 3
        assert args == (2, 1 + 3)
        assert kwargs == {'t': 1 + 4}

    def test_flatten(self):
        assert tuple(argmap._flatten([[[[[], []], [], []], [], [], []]], set())) == ()
        rlist = ['a', ['b', 'c'], [['d'], 'e'], 'f']
        assert ''.join(argmap._flatten(rlist, set())) == 'abcdef'

    def test_indent(self):
        code = '\n'.join(argmap._indent(*['try:', 'try:', 'pass#', 'finally:', 'pass#', '#', 'finally:', 'pass#']))
        assert code == 'try:\n try:\n  pass#\n finally:\n  pass#\n #\nfinally:\n pass#'

    def test_immediate_raise(self):

        @not_implemented_for('directed')
        def yield_nodes(G):
            yield from G
        G = nx.Graph([(1, 2)])
        D = nx.DiGraph()
        with pytest.raises(nx.NetworkXNotImplemented):
            node_iter = yield_nodes(D)
        with pytest.raises(nx.NetworkXNotImplemented):
            node_iter = yield_nodes(D)
        node_iter = yield_nodes(G)
        next(node_iter)
        next(node_iter)
        with pytest.raises(StopIteration):
            next(node_iter)