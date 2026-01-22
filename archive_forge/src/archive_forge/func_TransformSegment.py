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
def TransformSegment(r, index=-1, undefined=''):
    """Returns the index-th URI path segment.

  Args:
    r: A URI path.
    index: The path segment index to return counting from 0.
    undefined: Returns this value if the resource or segment index is empty.

  Returns:
    The index-th URI path segment in r
  """
    if not r:
        return undefined
    r = urllib.parse.unquote(six.text_type(r))
    segments = r.split('/')
    try:
        return segments[int(index)] or undefined
    except IndexError:
        return undefined