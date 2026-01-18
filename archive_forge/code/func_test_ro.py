import unittest
from zope.interface.common import ABCInterface
from zope.interface.common import ABCInterfaceClass
from zope.interface.verify import verifyClass
from zope.interface.verify import verifyObject
def test_ro(self, stdlib_class=stdlib_class, iface=iface):
    from zope.interface import Interface
    from zope.interface import implementedBy
    from zope.interface import ro
    self.assertEqual(tuple(ro.ro(iface, strict=True)), iface.__sro__)
    implements = implementedBy(stdlib_class)
    sro = implements.__sro__
    self.assertIs(sro[-1], Interface)
    if stdlib_class not in self.UNVERIFIABLE_RO:
        strict = stdlib_class not in self.NON_STRICT_RO
        isro = ro.ro(implements, strict=strict)
        isro.remove(Interface)
        isro.append(Interface)
        self.assertEqual(tuple(isro), sro)