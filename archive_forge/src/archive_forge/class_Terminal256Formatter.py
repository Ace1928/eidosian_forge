import sys
from pygments.formatter import Formatter
from pygments.console import codes
from pygments.style import ansicolors
class Terminal256Formatter(Formatter):
    """
    Format tokens with ANSI color sequences, for output in a 256-color
    terminal or console.  Like in `TerminalFormatter` color sequences
    are terminated at newlines, so that paging the output works correctly.

    The formatter takes colors from a style defined by the `style` option
    and converts them to nearest ANSI 256-color escape sequences. Bold and
    underline attributes from the style are preserved (and displayed).

    .. versionadded:: 0.9

    .. versionchanged:: 2.2
       If the used style defines foreground colors in the form ``#ansi*``, then
       `Terminal256Formatter` will map these to non extended foreground color.
       See :ref:`AnsiTerminalStyle` for more information.

    Options accepted:

    `style`
        The style to use, can be a string or a Style subclass (default:
        ``'default'``).
    """
    name = 'Terminal256'
    aliases = ['terminal256', 'console256', '256']
    filenames = []

    def __init__(self, **options):
        Formatter.__init__(self, **options)
        self.xterm_colors = []
        self.best_match = {}
        self.style_string = {}
        self.usebold = 'nobold' not in options
        self.useunderline = 'nounderline' not in options
        self._build_color_table()
        self._setup_styles()

    def _build_color_table(self):
        self.xterm_colors.append((0, 0, 0))
        self.xterm_colors.append((205, 0, 0))
        self.xterm_colors.append((0, 205, 0))
        self.xterm_colors.append((205, 205, 0))
        self.xterm_colors.append((0, 0, 238))
        self.xterm_colors.append((205, 0, 205))
        self.xterm_colors.append((0, 205, 205))
        self.xterm_colors.append((229, 229, 229))
        self.xterm_colors.append((127, 127, 127))
        self.xterm_colors.append((255, 0, 0))
        self.xterm_colors.append((0, 255, 0))
        self.xterm_colors.append((255, 255, 0))
        self.xterm_colors.append((92, 92, 255))
        self.xterm_colors.append((255, 0, 255))
        self.xterm_colors.append((0, 255, 255))
        self.xterm_colors.append((255, 255, 255))
        valuerange = (0, 95, 135, 175, 215, 255)
        for i in range(217):
            r = valuerange[i // 36 % 6]
            g = valuerange[i // 6 % 6]
            b = valuerange[i % 6]
            self.xterm_colors.append((r, g, b))
        for i in range(1, 22):
            v = 8 + i * 10
            self.xterm_colors.append((v, v, v))

    def _closest_color(self, r, g, b):
        distance = 257 * 257 * 3
        match = 0
        for i in range(0, 254):
            values = self.xterm_colors[i]
            rd = r - values[0]
            gd = g - values[1]
            bd = b - values[2]
            d = rd * rd + gd * gd + bd * bd
            if d < distance:
                match = i
                distance = d
        return match

    def _color_index(self, color):
        index = self.best_match.get(color, None)
        if color in ansicolors:
            index = color
            self.best_match[color] = index
        if index is None:
            try:
                rgb = int(str(color), 16)
            except ValueError:
                rgb = 0
            r = rgb >> 16 & 255
            g = rgb >> 8 & 255
            b = rgb & 255
            index = self._closest_color(r, g, b)
            self.best_match[color] = index
        return index

    def _setup_styles(self):
        for ttype, ndef in self.style:
            escape = EscapeSequence()
            if ndef['ansicolor']:
                escape.fg = self._color_index(ndef['ansicolor'])
            elif ndef['color']:
                escape.fg = self._color_index(ndef['color'])
            if ndef['bgansicolor']:
                escape.bg = self._color_index(ndef['bgansicolor'])
            elif ndef['bgcolor']:
                escape.bg = self._color_index(ndef['bgcolor'])
            if self.usebold and ndef['bold']:
                escape.bold = True
            if self.useunderline and ndef['underline']:
                escape.underline = True
            self.style_string[str(ttype)] = (escape.color_string(), escape.reset_string())

    def format(self, tokensource, outfile):
        if not self.encoding and hasattr(outfile, 'encoding') and hasattr(outfile, 'isatty') and outfile.isatty() and (sys.version_info < (3,)):
            self.encoding = outfile.encoding
        return Formatter.format(self, tokensource, outfile)

    def format_unencoded(self, tokensource, outfile):
        for ttype, value in tokensource:
            not_found = True
            while ttype and not_found:
                try:
                    on, off = self.style_string[str(ttype)]
                    spl = value.split('\n')
                    for line in spl[:-1]:
                        if line:
                            outfile.write(on + line + off)
                        outfile.write('\n')
                    if spl[-1]:
                        outfile.write(on + spl[-1] + off)
                    not_found = False
                except KeyError:
                    ttype = ttype[:-1]
            if not_found:
                outfile.write(value)