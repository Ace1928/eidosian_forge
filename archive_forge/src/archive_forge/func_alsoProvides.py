from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def alsoProvides(object, *interfaces):
    """
        Declare additional interfaces directly for an object.

        For example::

          alsoProvides(ob, I1)

        is equivalent to::

          directlyProvides(ob, directlyProvidedBy(ob), I1)

        .. seealso:: `zope.interface.alsoProvides`
        """