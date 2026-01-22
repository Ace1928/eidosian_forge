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
def TransformUpper(r):
    """Returns r in uppercase.

  Args:
    r: A resource key value.

  Returns:
    r in uppercase
  """
    if r and isinstance(r, six.string_types):
        return r.upper()
    return r