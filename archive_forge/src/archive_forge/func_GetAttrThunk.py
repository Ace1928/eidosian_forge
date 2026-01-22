from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import functools
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from six.moves import map  # pylint: disable=redefined-builtin
def GetAttrThunk(self, attr, transform=None):
    """Returns a thunk that gets the given attribute of the result of Get().

    Examples:

      >>> class A(object):
      ...   b = [1, 2, 3]
      >>> CachedResult([A()].pop).GetAttrThunk('b')()
      [1, 2, 3]
      >>> CachedResult([A()].pop).GetAttrThunk('b', lambda x: x+1)
      [2, 3, 4]

    Args:
      attr: str, the name of the attribute. Attribute should be iterable.
      transform: func, one-arg function that, if given, will be applied to
        every member of the attribute (which must be iterable) before returning
        it.

    Returns:
      zero-arg function which, when called, returns the attribute (possibly
        transformed) of the result (which is cached).
    """
    if transform:
        return lambda: list(map(transform, getattr(self.Get(), attr)))
    else:
        return lambda: getattr(self.Get(), attr)