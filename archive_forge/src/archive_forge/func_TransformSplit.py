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
def TransformSplit(r, sep='/', undefined=''):
    """Splits a string by the value of sep.

  Args:
    r: A string.
    sep: The separator value to use when splitting.
    undefined: Returns this value if the result after splitting is empty.

  Returns:
    A new array containing the split components of the resource.

  Example:
    `"a/b/c/d".split()` returns `["a", "b", "c", "d"]`.
  """
    if not r:
        return undefined
    try:
        return r.split(sep)
    except (AttributeError, TypeError, ValueError):
        return undefined