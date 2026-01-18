from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_proxyDeleteAttribute(self):
    """
        The attributes that proxy objects proxy should be deletable and affect
        the original object.
        """
    yayable = Yayable()
    yayable.ifaceAttribute = None
    proxy = proxyForInterface(IProxiedInterface)(yayable)
    del proxy.ifaceAttribute
    self.assertFalse(hasattr(yayable, 'ifaceAttribute'))