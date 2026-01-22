from zope.interface import classImplements
from zope.interface.common import collections
from zope.interface.common import io
from zope.interface.common import numbers
class INativeString(ITextString):
    """
    Interface for native strings.

    On all Python versions, this is :class:`str`. Tt extends
    :class:`ITextString`.
    """