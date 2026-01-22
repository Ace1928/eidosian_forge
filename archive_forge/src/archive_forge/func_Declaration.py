from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def Declaration(*interfaces):
    """
        Create an interface specification.

        The arguments are one or more interfaces or interface
        specifications (`IDeclaration` objects).

        A new interface specification (`IDeclaration`) with the given
        interfaces is returned.

        .. seealso:: `zope.interface.Declaration`
        """