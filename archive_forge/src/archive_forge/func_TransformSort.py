from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def TransformSort(r, attr=''):
    """Sorts the elements of the resource list by a given attribute (or itself).

  A string resource is treated as a list of characters.

  Args:
    r: A string or list.
    attr: The optional field of an object or dict by which to sort.

  Returns:
    A resource list ordered by the specified key.

  Example:
    `"b/a/d/c".split("/").sort()` returns `[a, b, c, d]`.
  """
    if not r:
        return []

    def SortKey(item):
        if not attr:
            return item
        return GetKeyValue(item, attr)
    return sorted(r, key=SortKey)