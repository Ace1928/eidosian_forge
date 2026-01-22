import sys
from pygments.formatter import Formatter
from pygments.console import codes
from pygments.style import ansicolors
def _setup_styles(self):
    for ttype, ndef in self.style:
        escape = EscapeSequence()
        if ndef['color']:
            escape.fg = self._color_tuple(ndef['color'])
        if ndef['bgcolor']:
            escape.bg = self._color_tuple(ndef['bgcolor'])
        if self.usebold and ndef['bold']:
            escape.bold = True
        if self.useunderline and ndef['underline']:
            escape.underline = True
        self.style_string[str(ttype)] = (escape.true_color_string(), escape.reset_string())