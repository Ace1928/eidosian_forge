import unittest
from zope.interface.common import ABCInterface
from zope.interface.common import ABCInterfaceClass
from zope.interface.verify import verifyClass
from zope.interface.verify import verifyObject
class VerifyClassMixin(unittest.TestCase):
    verifier = staticmethod(verifyClass)
    UNVERIFIABLE = ()
    NON_STRICT_RO = ()
    UNVERIFIABLE_RO = ()

    def _adjust_object_before_verify(self, iface, x):
        return x

    def verify(self, iface, klass, **kwargs):
        return self.verifier(iface, self._adjust_object_before_verify(iface, klass), **kwargs)