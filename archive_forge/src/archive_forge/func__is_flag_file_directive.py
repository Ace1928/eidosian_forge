from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def _is_flag_file_directive(self, flag_string):
    """Checks whether flag_string contain a --flagfile=<foo> directive."""
    if isinstance(flag_string, type('')):
        if flag_string.startswith('--flagfile='):
            return 1
        elif flag_string == '--flagfile':
            return 1
        elif flag_string.startswith('-flagfile='):
            return 1
        elif flag_string == '-flagfile':
            return 1
        else:
            return 0
    return 0