from __future__ import annotations
import copyreg
import io
import pickle
import sys
import textwrap
from typing import Any, Callable, List, Tuple
from typing_extensions import NoReturn
from twisted.persisted import aot, crefutil, styles
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
class AOTTests(TestCase):

    def test_simpleTypes(self) -> None:
        obj = (1, 2.0, 3j, True, slice(1, 2, 3), 'hello', 'world', sys.maxsize + 1, None, Ellipsis)
        rtObj = aot.unjellyFromSource(aot.jellyToSource(obj))
        self.assertEqual(obj, rtObj)

    def test_methodSelfIdentity(self) -> None:
        a = A()
        b = B()
        a.bmethod = b.bmethod
        b.a = a
        im_ = aot.unjellyFromSource(aot.jellyToSource(b)).a.bmethod
        self.assertEqual(aot._selfOfMethod(im_).__class__, aot._classOfMethod(im_))

    def test_methodNotSelfIdentity(self) -> None:
        """
        If a class change after an instance has been created,
        L{aot.unjellyFromSource} shoud raise a C{TypeError} when trying to
        unjelly the instance.
        """
        a = A()
        b = B()
        a.bmethod = b.bmethod
        b.a = a
        savedbmethod = B.bmethod
        del B.bmethod
        try:
            self.assertRaises(TypeError, aot.unjellyFromSource, aot.jellyToSource(b))
        finally:
            B.bmethod = savedbmethod

    def test_unsupportedType(self) -> None:
        """
        L{aot.jellyToSource} should raise a C{TypeError} when trying to jelly
        an unknown type without a C{__dict__} property or C{__getstate__}
        method.
        """

        class UnknownType:

            @property
            def __dict__(self) -> NoReturn:
                raise AttributeError()

            @property
            def __getstate__(self) -> NoReturn:
                raise AttributeError()
        self.assertRaises(TypeError, aot.jellyToSource, UnknownType())

    def test_basicIdentity(self) -> None:
        aj = aot.AOTJellier().jellyToAO
        d = {'hello': 'world', 'method': aj}
        l = [1, 2, 3, 'he\tllo\n\n"x world!', 'goodbye \n\tတ world!', 1, 1.0, 100 ** 100, unittest, aot.AOTJellier, d, funktion]
        t = tuple(l)
        l.append(l)
        l.append(t)
        l.append(t)
        uj = aot.unjellyFromSource(aot.jellyToSource([l, l]))
        assert uj[0] is uj[1]
        assert uj[1][0:5] == l[0:5]

    def test_nonDictState(self) -> None:
        a = NonDictState()
        a.state = 'meringue!'
        assert aot.unjellyFromSource(aot.jellyToSource(a)).state == a.state

    def test_copyReg(self) -> None:
        """
        L{aot.jellyToSource} and L{aot.unjellyFromSource} honor functions
        registered in the pickle copy registry.
        """
        uj = aot.unjellyFromSource(aot.jellyToSource(CopyRegistered()))
        self.assertIsInstance(uj, CopyRegisteredLoaded)

    def test_funkyReferences(self) -> None:
        o = EvilSourceror(EvilSourceror([]))
        j1 = aot.jellyToAOT(o)
        oj = aot.unjellyFromAOT(j1)
        assert oj.a is oj
        assert oj.a.b is oj.b
        assert oj.c is not oj.c.c

    def test_circularTuple(self) -> None:
        """
        L{aot.jellyToAOT} can persist circular references through tuples.
        """
        l: _CircularTupleType = []
        t = (l, 4321)
        l.append(t)
        j1 = aot.jellyToAOT(l)
        oj = aot.unjellyFromAOT(j1)
        self.assertIsInstance(oj[0], tuple)
        self.assertIs(oj[0][0], oj)
        self.assertEqual(oj[0][1], 4321)

    def testIndentify(self) -> None:
        """
        The generated serialization is indented.
        """
        self.assertEqual(aot.jellyToSource({'hello': {'world': []}}), textwrap.dedent("                app={\n                  'hello':{\n                    'world':[],\n                    },\n                  }"))