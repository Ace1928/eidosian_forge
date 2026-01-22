from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
class IProxiedSubInterface(IProxiedInterface):
    """
    An interface that derives from another for use with L{proxyForInterface}.
    """

    def boo():
        """
        A different sample method which should be proxied.
        """