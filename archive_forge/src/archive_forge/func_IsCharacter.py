from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def IsCharacter(self, characters, peek=False, eoi_ok=False):
    """Checks if the next character is in characters and consumes it if it is.

    Args:
      characters: A set of characters to check for. It may be a string, tuple,
        list or set.
      peek: Does not consume a matching character if True.
      eoi_ok: True if end of input is OK. Returns None if at end of input.

    Raises:
      ExpressionSyntaxError: End of input reached and peek and eoi_ok are False.

    Returns:
      The matching character or None if no match.
    """
    if self.EndOfInput():
        if peek or eoi_ok:
            return None
        raise resource_exceptions.ExpressionSyntaxError('More tokens expected [{0}].'.format(self.Annotate()))
    c = self._expr[self._position]
    if c not in characters:
        return None
    if not peek:
        self._position += 1
    return c