from zope.interface import classImplements
from zope.interface.common import collections
from zope.interface.common import io
from zope.interface.common import numbers
class IDict(collections.IMutableMapping):
    """
    Interface for :class:`dict`
    """
    extra_classes = (dict,)