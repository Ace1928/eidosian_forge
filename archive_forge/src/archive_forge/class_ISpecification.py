from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class ISpecification(Interface):
    """Object Behavioral specifications"""

    def providedBy(object):
        """Test whether the interface is implemented by the object

        Return true of the object asserts that it implements the
        interface, including asserting that it implements an extended
        interface.
        """

    def implementedBy(class_):
        """Test whether the interface is implemented by instances of the class

        Return true of the class asserts that its instances implement the
        interface, including asserting that they implement an extended
        interface.
        """

    def isOrExtends(other):
        """Test whether the specification is or extends another
        """

    def extends(other, strict=True):
        """Test whether a specification extends another

        The specification extends other if it has other as a base
        interface or if one of it's bases extends other.

        If strict is false, then the specification extends itself.
        """

    def weakref(callback=None):
        """Return a weakref to the specification

        This method is, regrettably, needed to allow weakrefs to be
        computed to security-proxied specifications.  While the
        zope.interface package does not require zope.security or
        zope.proxy, it has to be able to coexist with it.

        """
    __bases__ = Attribute('Base specifications\n\n    A tuple of specifications from which this specification is\n    directly derived.\n\n    ')
    __sro__ = Attribute("Specification-resolution order\n\n    A tuple of the specification and all of it's ancestor\n    specifications from most specific to least specific. The specification\n    itself is the first element.\n\n    (This is similar to the method-resolution order for new-style classes.)\n    ")
    __iro__ = Attribute("Interface-resolution order\n\n    A tuple of the specification's ancestor interfaces from\n    most specific to least specific.  The specification itself is\n    included if it is an interface.\n\n    (This is similar to the method-resolution order for new-style classes.)\n    ")

    def get(name, default=None):
        """Look up the description for a name

        If the named attribute is not defined, the default is
        returned.
        """