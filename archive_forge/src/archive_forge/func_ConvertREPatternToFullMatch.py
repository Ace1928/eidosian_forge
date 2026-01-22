from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.protorpclite import messages
from googlecloudsdk.core.resource import resource_expr_rewrite
from googlecloudsdk.core.util import times
import six
def ConvertREPatternToFullMatch(pattern, wordmatch=False):
    """Returns filter ~ pattern converted to a full match RE pattern.

  This function converts pattern such that the compute filter expression
    subject eq ConvertREPatternToFullMatch(pattern)
  matches (the entire subject matches) IFF
    re.search(pattern, subject)  # wordmatch=False
  matches (pattern matches anywhere in subject).

  Args:
    pattern: A RE pattern that partially matches the subject string.
    wordmatch: True if ^ and $ anchors should be converted to word boundaries.

  Returns:
    The converted ~ pattern suitable for the compute eq filter match operator.
  """
    if wordmatch:
        cclass = 0
        escape = False
        full = []
        for c in pattern:
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif cclass:
                if c == ']':
                    if cclass == 1:
                        cclass = 2
                    else:
                        cclass = 0
                elif c != '^':
                    cclass = 2
            elif c == '[':
                cclass = 1
            elif c in ('^', '$'):
                c = '\\b'
            full.append(c)
        pattern = ''.join(full)
    return '".*(' + pattern.replace('"', '\\"') + ').*"'