import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class MethodTests(AttributeTests):
    DEFAULT_NAME = 'TestMethod'

    def _getTargetClass(self):
        from zope.interface.interface import Method
        return Method

    def test_optional_as_property(self):
        method = self._makeOne()
        self.assertEqual(method.optional, {})
        method.optional = {'foo': 'bar'}
        self.assertEqual(method.optional, {'foo': 'bar'})
        del method.optional
        self.assertEqual(method.optional, {})

    def test___call___raises_BrokenImplementation(self):
        from zope.interface.exceptions import BrokenImplementation
        method = self._makeOne()
        try:
            method()
        except BrokenImplementation as e:
            self.assertEqual(e.interface, None)
            self.assertEqual(e.name, self.DEFAULT_NAME)
        else:
            self.fail('__call__ should raise BrokenImplementation')

    def test_getSignatureInfo_bare(self):
        method = self._makeOne()
        info = method.getSignatureInfo()
        self.assertEqual(list(info['positional']), [])
        self.assertEqual(list(info['required']), [])
        self.assertEqual(info['optional'], {})
        self.assertEqual(info['varargs'], None)
        self.assertEqual(info['kwargs'], None)

    def test_getSignatureString_bare(self):
        method = self._makeOne()
        self.assertEqual(method.getSignatureString(), '()')

    def test_getSignatureString_w_only_required(self):
        method = self._makeOne()
        method.positional = method.required = ['foo']
        self.assertEqual(method.getSignatureString(), '(foo)')

    def test_getSignatureString_w_optional(self):
        method = self._makeOne()
        method.positional = method.required = ['foo']
        method.optional = {'foo': 'bar'}
        self.assertEqual(method.getSignatureString(), "(foo='bar')")

    def test_getSignatureString_w_varargs(self):
        method = self._makeOne()
        method.varargs = 'args'
        self.assertEqual(method.getSignatureString(), '(*args)')

    def test_getSignatureString_w_kwargs(self):
        method = self._makeOne()
        method.kwargs = 'kw'
        self.assertEqual(method.getSignatureString(), '(**kw)')

    def test__repr__w_interface(self):
        method = self._makeOne()
        method.kwargs = 'kw'
        method.interface = type(self)
        r = repr(method)
        self.assertTrue(r.startswith('<zope.interface.interface.Method object at'), r)
        self.assertTrue(r.endswith(' ' + __name__ + '.MethodTests.TestMethod(**kw)>'), r)

    def test__repr__wo_interface(self):
        method = self._makeOne()
        method.kwargs = 'kw'
        r = repr(method)
        self.assertTrue(r.startswith('<zope.interface.interface.Method object at'), r)
        self.assertTrue(r.endswith(' TestMethod(**kw)>'), r)

    def test__str__w_interface(self):
        method = self._makeOne()
        method.kwargs = 'kw'
        method.interface = type(self)
        r = str(method)
        self.assertEqual(r, __name__ + '.MethodTests.TestMethod(**kw)')

    def test__str__wo_interface(self):
        method = self._makeOne()
        method.kwargs = 'kw'
        r = str(method)
        self.assertEqual(r, 'TestMethod(**kw)')