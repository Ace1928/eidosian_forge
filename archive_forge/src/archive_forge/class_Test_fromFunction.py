import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class Test_fromFunction(unittest.TestCase):

    def _callFUT(self, *args, **kw):
        from zope.interface.interface import fromFunction
        return fromFunction(*args, **kw)

    def test_bare(self):

        def _func():
            """DOCSTRING"""
        method = self._callFUT(_func)
        self.assertEqual(method.getName(), '_func')
        self.assertEqual(method.getDoc(), 'DOCSTRING')
        self.assertEqual(method.interface, None)
        self.assertEqual(list(method.getTaggedValueTags()), [])
        info = method.getSignatureInfo()
        self.assertEqual(list(info['positional']), [])
        self.assertEqual(list(info['required']), [])
        self.assertEqual(info['optional'], {})
        self.assertEqual(info['varargs'], None)
        self.assertEqual(info['kwargs'], None)

    def test_w_interface(self):
        from zope.interface.interface import InterfaceClass

        class IFoo(InterfaceClass):
            pass

        def _func():
            """DOCSTRING"""
        method = self._callFUT(_func, interface=IFoo)
        self.assertEqual(method.interface, IFoo)

    def test_w_name(self):

        def _func():
            """DOCSTRING"""
        method = self._callFUT(_func, name='anotherName')
        self.assertEqual(method.getName(), 'anotherName')

    def test_w_only_required(self):

        def _func(foo):
            """DOCSTRING"""
        method = self._callFUT(_func)
        info = method.getSignatureInfo()
        self.assertEqual(list(info['positional']), ['foo'])
        self.assertEqual(list(info['required']), ['foo'])
        self.assertEqual(info['optional'], {})
        self.assertEqual(info['varargs'], None)
        self.assertEqual(info['kwargs'], None)

    def test_w_optional(self):

        def _func(foo='bar'):
            """DOCSTRING"""
        method = self._callFUT(_func)
        info = method.getSignatureInfo()
        self.assertEqual(list(info['positional']), ['foo'])
        self.assertEqual(list(info['required']), [])
        self.assertEqual(info['optional'], {'foo': 'bar'})
        self.assertEqual(info['varargs'], None)
        self.assertEqual(info['kwargs'], None)

    def test_w_optional_self(self):

        def _func(self='bar'):
            """DOCSTRING"""
        method = self._callFUT(_func, imlevel=1)
        info = method.getSignatureInfo()
        self.assertEqual(list(info['positional']), [])
        self.assertEqual(list(info['required']), [])
        self.assertEqual(info['optional'], {})
        self.assertEqual(info['varargs'], None)
        self.assertEqual(info['kwargs'], None)

    def test_w_varargs(self):

        def _func(*args):
            """DOCSTRING"""
        method = self._callFUT(_func)
        info = method.getSignatureInfo()
        self.assertEqual(list(info['positional']), [])
        self.assertEqual(list(info['required']), [])
        self.assertEqual(info['optional'], {})
        self.assertEqual(info['varargs'], 'args')
        self.assertEqual(info['kwargs'], None)

    def test_w_kwargs(self):

        def _func(**kw):
            """DOCSTRING"""
        method = self._callFUT(_func)
        info = method.getSignatureInfo()
        self.assertEqual(list(info['positional']), [])
        self.assertEqual(list(info['required']), [])
        self.assertEqual(info['optional'], {})
        self.assertEqual(info['varargs'], None)
        self.assertEqual(info['kwargs'], 'kw')

    def test_full_spectrum(self):

        def _func(foo, bar='baz', *args, **kw):
            """DOCSTRING"""
        method = self._callFUT(_func)
        info = method.getSignatureInfo()
        self.assertEqual(list(info['positional']), ['foo', 'bar'])
        self.assertEqual(list(info['required']), ['foo'])
        self.assertEqual(info['optional'], {'bar': 'baz'})
        self.assertEqual(info['varargs'], 'args')
        self.assertEqual(info['kwargs'], 'kw')