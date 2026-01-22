from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_transform
import six
def _ParseKeys(self):
    """Parses a comma separated list of keys.

    The initial '(' has already been consumed by the caller.

    Raises:
      ExpressionSyntaxError: The expression has a syntax error.
    """
    if self._lex.IsCharacter(')'):
        return
    while True:
        self._ParseKey()
        self._lex.SkipSpace()
        if self._lex.IsCharacter(')'):
            break
        if not self._lex.IsCharacter(','):
            raise resource_exceptions.ExpressionSyntaxError('Expected ) in projection expression [{0}].'.format(self._lex.Annotate()))