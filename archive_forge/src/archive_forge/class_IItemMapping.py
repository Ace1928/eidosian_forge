from zope.interface import Interface
from zope.interface.common import collections
class IItemMapping(Interface):
    """Simplest readable mapping object
    """

    def __getitem__(key):
        """Get a value for a key

        A `KeyError` is raised if there is no value for the key.
        """