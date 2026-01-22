from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.protorpclite import messages
from googlecloudsdk.core.resource import resource_expr_rewrite
from googlecloudsdk.core.util import times
import six
def ConvertEQPatternToFullMatch(pattern):
    """Returns filter = pattern converted to a full match RE pattern.

  This function converts pattern such that the compute filter expression
    subject eq ConvertEQPatternToFullMatch(pattern)
  matches (the entire subject matches) IFF
    re.search(r'\\b' + _EscapePattern(pattern) + r'\\b', subject)
  matches (pattern matches anywhere in subject).

  Args:
    pattern: A filter = pattern that partially matches the subject string.

  Returns:
    The converted = pattern suitable for the compute eq filter match operator.
  """
    return '".*\\b{pattern}\\b.*"'.format(pattern=_EscapePattern(pattern))