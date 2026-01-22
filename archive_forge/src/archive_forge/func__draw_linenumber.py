import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _draw_linenumber(self, posno, lineno):
    """
        Remember a line number drawable to paint later.
        """
    self._draw_text(self._get_linenumber_pos(posno), str(lineno).rjust(self.line_number_chars), font=self.fonts.get_font(self.line_number_bold, self.line_number_italic), fill=self.line_number_fg)