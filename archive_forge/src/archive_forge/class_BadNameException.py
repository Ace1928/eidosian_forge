from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import six.moves.urllib.parse
class BadNameException(Exception):
    """Exceptions when a bad docker name is supplied."""