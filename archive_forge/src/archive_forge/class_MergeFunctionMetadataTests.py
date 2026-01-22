import errno
import os.path
import shutil
import sys
import warnings
from typing import Iterable, Mapping, MutableMapping, Sequence
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.error import ProcessDone
from twisted.internet.interfaces import IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.python import util
from twisted.python.filepath import FilePath
from twisted.test.test_process import MockOS
from twisted.trial.unittest import FailTest, TestCase
from twisted.trial.util import suppress as SUPPRESS
class MergeFunctionMetadataTests(TestCase):
    """
    Tests for L{mergeFunctionMetadata}.
    """

    def test_mergedFunctionBehavesLikeMergeTarget(self):
        """
        After merging C{foo}'s data into C{bar}, the returned function behaves
        as if it is C{bar}.
        """
        foo_object = object()
        bar_object = object()

        def foo():
            return foo_object

        def bar(x, y, ab, c=10, *d, **e):
            a, b = ab
            return bar_object
        baz = util.mergeFunctionMetadata(foo, bar)
        self.assertIs(baz(1, 2, (3, 4), quux=10), bar_object)

    def test_moduleIsMerged(self):
        """
        Merging C{foo} into C{bar} returns a function with C{foo}'s
        C{__module__}.
        """

        def foo():
            pass

        def bar():
            pass
        bar.__module__ = 'somewhere.else'
        baz = util.mergeFunctionMetadata(foo, bar)
        self.assertEqual(baz.__module__, foo.__module__)

    def test_docstringIsMerged(self):
        """
        Merging C{foo} into C{bar} returns a function with C{foo}'s docstring.
        """

        def foo():
            """
            This is foo.
            """

        def bar():
            """
            This is bar.
            """
        baz = util.mergeFunctionMetadata(foo, bar)
        self.assertEqual(baz.__doc__, foo.__doc__)

    def test_nameIsMerged(self):
        """
        Merging C{foo} into C{bar} returns a function with C{foo}'s name.
        """

        def foo():
            pass

        def bar():
            pass
        baz = util.mergeFunctionMetadata(foo, bar)
        self.assertEqual(baz.__name__, foo.__name__)

    def test_instanceDictionaryIsMerged(self):
        """
        Merging C{foo} into C{bar} returns a function with C{bar}'s
        dictionary, updated by C{foo}'s.
        """

        def foo():
            pass
        foo.a = 1
        foo.b = 2

        def bar():
            pass
        bar.b = 3
        bar.c = 4
        baz = util.mergeFunctionMetadata(foo, bar)
        self.assertEqual(foo.a, baz.a)
        self.assertEqual(foo.b, baz.b)
        self.assertEqual(bar.c, baz.c)