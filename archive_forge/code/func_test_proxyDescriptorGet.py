from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_proxyDescriptorGet(self):
    """
        _ProxyDescriptor's __get__ method should return the appropriate
        attribute of its argument's 'original' attribute if it is invoked with
        an object.  If it is invoked with None, it should return a false
        class-method emulator instead.

        For some reason, Python's documentation recommends to define
        descriptors' __get__ methods with the 'type' parameter as optional,
        despite the fact that Python itself never actually calls the descriptor
        that way.  This is probably do to support 'foo.__get__(bar)' as an
        idiom.  Let's make sure that the behavior is correct.  Since we don't
        actually use the 'type' argument at all, this test calls it the
        idiomatic way to ensure that signature works; test_proxyInheritance
        verifies the how-Python-actually-calls-it signature.
        """

    class Sample:
        called = False

        def hello(self):
            self.called = True
    fakeProxy = Sample()
    testObject = Sample()
    fakeProxy.original = testObject
    pd = components._ProxyDescriptor('hello', 'original')
    self.assertEqual(pd.__get__(fakeProxy), testObject.hello)
    fakeClassMethod = pd.__get__(None)
    fakeClassMethod(fakeProxy)
    self.assertTrue(testObject.called)