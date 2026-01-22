import sys
from pygments.formatter import Formatter
from pygments.console import codes
from pygments.style import ansicolors
class EscapeSequence:

    def __init__(self, fg=None, bg=None, bold=False, underline=False):
        self.fg = fg
        self.bg = bg
        self.bold = bold
        self.underline = underline

    def escape(self, attrs):
        if len(attrs):
            return '\x1b[' + ';'.join(attrs) + 'm'
        return ''

    def color_string(self):
        attrs = []
        if self.fg is not None:
            if self.fg in ansicolors:
                esc = codes[self.fg[5:]]
                if ';01m' in esc:
                    self.bold = True
                attrs.append(esc[2:4])
            else:
                attrs.extend(('38', '5', '%i' % self.fg))
        if self.bg is not None:
            if self.bg in ansicolors:
                esc = codes[self.bg[5:]]
                attrs.append(str(int(esc[2:4]) + 10))
            else:
                attrs.extend(('48', '5', '%i' % self.bg))
        if self.bold:
            attrs.append('01')
        if self.underline:
            attrs.append('04')
        return self.escape(attrs)

    def true_color_string(self):
        attrs = []
        if self.fg:
            attrs.extend(('38', '2', str(self.fg[0]), str(self.fg[1]), str(self.fg[2])))
        if self.bg:
            attrs.extend(('48', '2', str(self.bg[0]), str(self.bg[1]), str(self.bg[2])))
        if self.bold:
            attrs.append('01')
        if self.underline:
            attrs.append('04')
        return self.escape(attrs)

    def reset_string(self):
        attrs = []
        if self.fg is not None:
            attrs.append('39')
        if self.bg is not None:
            attrs.append('49')
        if self.bold or self.underline:
            attrs.append('00')
        return self.escape(attrs)