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
def TransformSub(r, pattern, replacement, count=0, ignorecase=True):
    """Replaces a pattern matched in a string with the given replacement.

  Return the string obtained by replacing the leftmost non-overlapping
  occurrences of pattern in the string by replacement. If the pattern isn't
  found, then the original string is returned unchanged.

  Args:
    r: A string
    pattern: The regular expression pattern to match in r that we want to
      replace with something.
    replacement: The value to substitute into whatever pattern is matched.
    count: The max number of pattern occurrences to be replaced. Must be
      non-negative. If omitted or zero, all occurrences will be replaces.
    ignorecase: Whether to perform case-insensitive matching.

  Returns:
    A new string with the replacements applied.

  Example:
    `table(field.sub(" there", ""))`:::
    If the field string is "hey there" it will be displayed as "hey".
  """
    try:
        count = int(count)
    except ValueError:
        return r
    try:
        ignorecase = re.IGNORECASE if GetBooleanArgValue(ignorecase) else 0
        flags = re.MULTILINE | re.DOTALL | ignorecase
        return re.sub(pattern, replacement, r, count, flags)
    except re.error:
        return r