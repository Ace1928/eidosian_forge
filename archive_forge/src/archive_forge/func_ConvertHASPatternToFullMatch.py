from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.protorpclite import messages
from googlecloudsdk.core.resource import resource_expr_rewrite
from googlecloudsdk.core.util import times
import six
def ConvertHASPatternToFullMatch(pattern):
    """Returns filter : pattern converted to a full match RE pattern.

  This function converts pattern such that the compute filter expression
    subject eq ConvertREPatternToFullMatch(pattern)
  matches (the entire subject matches) IFF
    re.search(r'\\b' + _EscapePattern(pattern) + r'\\b', subject)  # no trailing *
    re.search(r'\\b' + _EscapePattern(pattern[:-1]), subject)     # trailing *
  matches (pattern matches anywhere in subject).

  Args:
    pattern: A filter : pattern that partially matches the subject string.

  Returns:
    The converted : pattern suitable for the compute eq filter match operator.
  """
    left = '.*\\b'
    if pattern.endswith('*'):
        pattern = pattern[:-1]
        right = '.*'
    else:
        right = '\\b.*'
    return '"{left}{pattern}{right}"'.format(left=left, pattern=_EscapePattern(pattern), right=right)