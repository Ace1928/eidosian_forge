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
def MergeLogExpressions(log_format, expressions):
    """Replaces each $N substring with the corresponding {expression}.

  This function is intended for reconstructing an input expression string that
  has been split using SplitLogExpressions. It is not intended for substituting
  the expression results at log time.

  Args:
    log_format: A string containing 0 or more $N substrings, where N is any
      valid index into the expressions array. Each such substring will be
      replaced by '{expression}', where "expression" is expressions[N].
    expressions: The expressions to substitute into the format string.
  Returns:
    The combined string.
  """

    def GetExpression(m):
        try:
            return '{{{0}}}'.format(expressions[int(m.group(0)[1:])])
        except IndexError:
            return m.group(0)
    parts = log_format.split('$$')
    return '$'.join((re.sub('\\$\\d+', GetExpression, part) for part in parts))