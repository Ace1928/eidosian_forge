from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from absl.flags import _helpers
class IllegalFlagValueError(Error):
    """Raised when the flag command line argument is illegal."""