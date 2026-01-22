import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class Test_fromMethod(unittest.TestCase):

    def _callFUT(self, *args, **kw):
        from zope.interface.interface import fromMethod
        return fromMethod(*args, **kw)

    def test_no_args(self):

        class Foo:

            def bar(self):
                """DOCSTRING"""
        method = self._callFUT(Foo.bar)
        self.assertEqual(method.getName(), 'bar')
        self.assertEqual(method.getDoc(), 'DOCSTRING')
        self.assertEqual(method.interface, None)
        self.assertEqual(list(method.getTaggedValueTags()), [])
        info = method.getSignatureInfo()
        self.assertEqual(list(info['positional']), [])
        self.assertEqual(list(info['required']), [])
        self.assertEqual(info['optional'], {})
        self.assertEqual(info['varargs'], None)
        self.assertEqual(info['kwargs'], None)

    def test_full_spectrum(self):

        class Foo:

            def bar(self, foo, bar='baz', *args, **kw):
                """DOCSTRING"""
        method = self._callFUT(Foo.bar)
        info = method.getSignatureInfo()
        self.assertEqual(list(info['positional']), ['foo', 'bar'])
        self.assertEqual(list(info['required']), ['foo'])
        self.assertEqual(info['optional'], {'bar': 'baz'})
        self.assertEqual(info['varargs'], 'args')
        self.assertEqual(info['kwargs'], 'kw')

    def test_w_non_method(self):

        def foo():
            """DOCSTRING"""
        method = self._callFUT(foo)
        self.assertEqual(method.getName(), 'foo')
        self.assertEqual(method.getDoc(), 'DOCSTRING')
        self.assertEqual(method.interface, None)
        self.assertEqual(list(method.getTaggedValueTags()), [])
        info = method.getSignatureInfo()
        self.assertEqual(list(info['positional']), [])
        self.assertEqual(list(info['required']), [])
        self.assertEqual(info['optional'], {})
        self.assertEqual(info['varargs'], None)
        self.assertEqual(info['kwargs'], None)