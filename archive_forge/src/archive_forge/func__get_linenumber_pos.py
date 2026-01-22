import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _get_linenumber_pos(self, lineno):
    """
        Get the actual position for the start of a line number.
        """
    return (self.image_pad, self._get_line_y(lineno))