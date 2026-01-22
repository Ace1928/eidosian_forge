from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IDeclaration(ISpecification):
    """Interface declaration

    Declarations are used to express the interfaces implemented by
    classes or provided by objects.
    """

    def __contains__(interface):
        """Test whether an interface is in the specification

        Return true if the given interface is one of the interfaces in
        the specification and false otherwise.
        """

    def __iter__():
        """Return an iterator for the interfaces in the specification
        """

    def flattened():
        """Return an iterator of all included and extended interfaces

        An iterator is returned for all interfaces either included in
        or extended by interfaces included in the specifications
        without duplicates. The interfaces are in "interface
        resolution order". The interface resolution order is such that
        base interfaces are listed after interfaces that extend them
        and, otherwise, interfaces are included in the order that they
        were defined in the specification.
        """

    def __sub__(interfaces):
        """Create an interface specification with some interfaces excluded

        The argument can be an interface or an interface
        specifications.  The interface or interfaces given in a
        specification are subtracted from the interface specification.

        Removing an interface that is not in the specification does
        not raise an error. Doing so has no effect.

        Removing an interface also removes sub-interfaces of the interface.

        """

    def __add__(interfaces):
        """Create an interface specification with some interfaces added

        The argument can be an interface or an interface
        specifications.  The interface or interfaces given in a
        specification are added to the interface specification.

        Adding an interface that is already in the specification does
        not raise an error. Doing so has no effect.
        """

    def __nonzero__():
        """Return a true value of the interface specification is non-empty
        """