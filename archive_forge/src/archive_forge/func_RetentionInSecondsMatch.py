from __future__ import absolute_import
from six.moves import input
from decimal import Decimal
import re
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def RetentionInSecondsMatch(seconds):
    """Test whether the string matches retention in seconds pattern.

  Args:
    seconds: string to match for retention specified in seconds format.

  Returns:
    Returns a match object if the string matches the retention in seconds
    pattern. The match object will contain a 'number' group for the duration
    in number of seconds. Otherwise, None is returned.
  """
    return _RETENTION_IN_SECONDS().match(seconds)