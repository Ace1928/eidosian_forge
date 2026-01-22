from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
def absl_to_standard(level):
    """Converts an integer level from the absl value to the standard value.

  Args:
    level: int, an absl.logging level.

  Raises:
    TypeError: Raised when level is not an integer.

  Returns:
    The corresponding integer level for use in standard logging.
  """
    if not isinstance(level, int):
        raise TypeError('Expect an int level, found {}'.format(type(level)))
    if level < ABSL_FATAL:
        level = ABSL_FATAL
    if level <= ABSL_DEBUG:
        return ABSL_TO_STANDARD[level]
    return STANDARD_DEBUG - level + 1