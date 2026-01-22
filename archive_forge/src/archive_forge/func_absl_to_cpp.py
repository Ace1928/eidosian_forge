from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
def absl_to_cpp(level):
    """Converts an absl log level to a cpp log level.

  Args:
    level: int, an absl.logging level.

  Raises:
    TypeError: Raised when level is not an integer.

  Returns:
    The corresponding integer level for use in Abseil C++.
  """
    if not isinstance(level, int):
        raise TypeError('Expect an int level, found {}'.format(type(level)))
    if level >= 0:
        return 0
    else:
        return -level