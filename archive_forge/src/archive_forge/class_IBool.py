from zope.interface import classImplements
from zope.interface.common import collections
from zope.interface.common import io
from zope.interface.common import numbers
class IBool(numbers.IIntegral):
    """
    Interface for :class:`bool`
    """
    extra_classes = (bool,)