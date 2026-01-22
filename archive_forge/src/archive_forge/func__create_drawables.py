import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _create_drawables(self, tokensource):
    """
        Create drawables for the token content.
        """
    lineno = charno = maxcharno = 0
    for ttype, value in tokensource:
        while ttype not in self.styles:
            ttype = ttype.parent
        style = self.styles[ttype]
        value = value.expandtabs(4)
        lines = value.splitlines(True)
        for i, line in enumerate(lines):
            temp = line.rstrip('\n')
            if temp:
                self._draw_text(self._get_text_pos(charno, lineno), temp, font=self._get_style_font(style), fill=self._get_text_color(style))
                charno += len(temp)
                maxcharno = max(maxcharno, charno)
            if line.endswith('\n'):
                charno = 0
                lineno += 1
    self.maxcharno = maxcharno
    self.maxlineno = lineno