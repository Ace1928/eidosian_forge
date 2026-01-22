import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _get_char_width(self):
    """
        Get the width of a character.
        """
    return self.fontw