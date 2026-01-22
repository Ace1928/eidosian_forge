from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
import unicodedata
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
import six
def _ReCompile(pattern, flags=0):
    """Returns a compiled RE pattern.

  Args:
    pattern: The RE pattern string.
    flags: Optional RE flags.

  Raises:
    ExpressionSyntaxError: RE pattern error.

  Returns:
    The compiled RE.
  """
    try:
        return re.compile(pattern, flags)
    except re.error as e:
        raise resource_exceptions.ExpressionSyntaxError('Filter expression RE pattern [{}]: {}'.format(pattern, e))