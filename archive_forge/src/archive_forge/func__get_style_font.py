import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _get_style_font(self, style):
    """
        Get the correct font for the style.
        """
    return self.fonts.get_font(style['bold'], style['italic'])