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
def EndOfInput(self, position=None):
    """Checks if the current expression string position is at the end of input.

    Args:
      position: Checks position instead of the current expression position.

    Returns:
      True if the expression string position is at the end of input.
    """
    if position is None:
        position = self._position
    return position >= len(self._expr)