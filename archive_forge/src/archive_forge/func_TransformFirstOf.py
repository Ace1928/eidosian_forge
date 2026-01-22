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
def TransformFirstOf(r, *keys):
    """Returns the first non-empty attribute value for key in keys.

  Args:
    r: A JSON-serializable object.
    *keys: Keys to check for resource attribute values,

  Returns:
    The first non-empty r.key value for key in keys, '' otherwise.

  Example:
    `x.firstof(bar_foo, barFoo, BarFoo, BAR_FOO)`:::
    Checks x.bar_foo, x.barFoo, x.BarFoo, and x.BAR_FOO in order for the first
    non-empty value.
  """
    for key in keys:
        v = GetKeyValue(r, key)
        if v is not None:
            return v
    return ''