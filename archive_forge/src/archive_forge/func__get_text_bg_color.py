import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _get_text_bg_color(self, style):
    """
        Get the correct background color for the token from the style.
        """
    if style['bgcolor'] is not None:
        bg_color = '#' + style['bgcolor']
    else:
        bg_color = None
    return bg_color