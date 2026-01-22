from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from absl.flags import _helpers
class CantOpenFlagFileError(Error):
    """Raised when flagfile fails to open.

  E.g. the file doesn't exist, or has wrong permissions.
  """