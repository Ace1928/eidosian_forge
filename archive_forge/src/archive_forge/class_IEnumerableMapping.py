from zope.interface import Interface
from zope.interface.common import collections
class IEnumerableMapping(collections.ISized, IReadMapping):
    """
    Mapping objects whose items can be enumerated.

    .. versionchanged:: 5.0.0
       Extend ``ISized``
    """

    def keys():
        """Return the keys of the mapping object.
        """

    def __iter__():
        """Return an iterator for the keys of the mapping object.
        """

    def values():
        """Return the values of the mapping object.
        """

    def items():
        """Return the items of the mapping object.
        """