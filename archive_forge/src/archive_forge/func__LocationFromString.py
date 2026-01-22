from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import threading
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.debug import errors
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
import six
from six.moves import urllib
def _LocationFromString(self, location):
    """Converts a file:line location string into a SourceLocation.

    Args:
      location: A string of the form file:line.
    Returns:
      The corresponding SourceLocation message.
    Raises:
      InvalidLocationException: if the line is not of the form path:line
    """
    components = location.split(':')
    if len(components) != 2:
        raise errors.InvalidLocationException('Location must be of the form "path:line"')
    try:
        return self._debug_messages.SourceLocation(path=components[0], line=int(components[1]))
    except ValueError:
        raise errors.InvalidLocationException('Location must be of the form "path:line", where "line" must be an integer.')