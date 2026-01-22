import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _get_line_height(self):
    """
        Get the height of a line.
        """
    return self.fonth + self.line_pad