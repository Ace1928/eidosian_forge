from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import inspect
from fire import inspectutils
import six
def _FormatForCommand(token):
    """Replaces underscores with hyphens, unless the token starts with a token.

  This is because we typically prefer hyphens to underscores at the command
  line, but we reserve hyphens at the start of a token for flags. This becomes
  relevant when --verbose is activated, so that things like __str__ don't get
  transformed into --str--, which would get confused for a flag.

  Args:
    token: The token to transform.
  Returns:
    The transformed token.
  """
    if not isinstance(token, six.string_types):
        token = str(token)
    if token.startswith('_'):
        return token
    return token.replace('_', '-')