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
def TransformBaseName(r, undefined=''):
    """Returns the last path component.

  Args:
    r: A URI or unix/windows file path.
    undefined: Returns this value if the resource or basename is empty.

  Returns:
    The last path component.
  """
    if not r:
        return undefined
    if isinstance(r, list):
        return [TransformBaseName(i) for i in r]
    s = six.text_type(r)
    for separator in ('/', '\\'):
        i = s.rfind(separator)
        if i >= 0:
            return s[i + 1:]
    return s or undefined