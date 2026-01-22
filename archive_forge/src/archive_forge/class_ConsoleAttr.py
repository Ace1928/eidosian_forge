from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import locale
import os
import sys
import unicodedata
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import encoding as encoding_util
import six
class ConsoleAttr(object):
    """Console attribute and special drawing characters and functions accessor.

  Use GetConsoleAttr() to get a global ConsoleAttr object shared by all callers.
  Use ConsoleAttr() for abstracting multiple consoles.

  If _out is not associated with a console, or if the console properties cannot
  be determined, the default behavior is ASCII art with no attributes.

  Attributes:
    _ANSI_COLOR: The ANSI color control sequence dict.
    _ANSI_COLOR_RESET: The ANSI color reset control sequence string.
    _csi: The ANSI Control Sequence indicator string, '' if not supported.
    _encoding: The character encoding.
        ascii: ASCII art. This is the default.
        utf8: UTF-8 unicode.
        win: Windows code page 437.
    _font_bold: The ANSI bold font embellishment code string.
    _font_italic: The ANSI italic font embellishment code string.
    _get_raw_key: A function that reads one keypress from stdin with no echo.
    _out: The console output file stream.
    _term: TERM environment variable value.
    _term_size: The terminal (x, y) dimensions in characters.
  """
    _CONSOLE_ATTR_STATE = None
    _ANSI_COLOR = {'red': '31;1m', 'yellow': '33;1m', 'green': '32m', 'blue': '34;1m'}
    _ANSI_COLOR_RESET = '39;0m'
    _BULLETS_UNICODE = ('▪', '◆', '▸', '▫', '◇', '▹')
    _BULLETS_WINDOWS = ('■', '≡', '∞', 'Φ', '·')
    _BULLETS_ASCII = ('o', '*', '+', '-')

    def __init__(self, encoding=None, term=None, suppress_output=False):
        """Constructor.

    Args:
      encoding: Encoding override.
        ascii -- ASCII art. This is the default.
        utf8 -- UTF-8 unicode.
        win -- Windows code page 437.
      term: Terminal override. Replaces the value of ENV['TERM'].
      suppress_output: True to create a ConsoleAttr that doesn't want to output
        anything.
    """
        if not encoding:
            encoding = self._GetConsoleEncoding()
        elif encoding == 'win':
            encoding = 'cp437'
        self._encoding = encoding or 'ascii'
        if suppress_output:
            self._term = ''
        elif term:
            self._term = term
        else:
            self._term = encoding_util.GetEncodedValue(os.environ, 'TERM', '').lower()
        if self.SupportsAnsi():
            self._csi = '\x1b['
            self._font_bold = '1'
            self._font_italic = '4'
        else:
            self._csi = None
            self._font_bold = ''
            self._font_italic = ''
        is_screen_reader = properties.VALUES.accessibility.screen_reader.GetBool()
        if self._encoding == 'utf-8' and (not is_screen_reader):
            self._box_line_characters = BoxLineCharactersUnicode()
            self._bullets = self._BULLETS_UNICODE
            self._progress_tracker_symbols = ProgressTrackerSymbolsUnicode()
        elif self._encoding == 'cp437' and (not is_screen_reader):
            self._box_line_characters = BoxLineCharactersUnicode()
            self._bullets = self._BULLETS_WINDOWS
            self._progress_tracker_symbols = ProgressTrackerSymbolsAscii()
        else:
            self._box_line_characters = BoxLineCharactersAscii()
            if is_screen_reader:
                self._box_line_characters = BoxLineCharactersScreenReader()
            self._bullets = self._BULLETS_ASCII
            self._progress_tracker_symbols = ProgressTrackerSymbolsAscii()
        self._get_raw_key = [console_attr_os.GetRawKeyFunction()]
        self._term_size = (0, 0) if suppress_output else console_attr_os.GetTermSize()
        self._display_width_cache = {}

    def _GetConsoleEncoding(self):
        """Gets the encoding as declared by the stdout stream.

    Returns:
      str, The encoding name or None if it could not be determined.
    """
        console_encoding = getattr(sys.stdout, 'encoding', None)
        if not console_encoding:
            return None
        console_encoding = console_encoding.lower()
        if 'utf-8' in console_encoding:
            locale_encoding = locale.getpreferredencoding()
            if locale_encoding and 'cp1252' in locale_encoding:
                return None
            return 'utf-8'
        elif 'cp437' in console_encoding:
            return 'cp437'
        elif 'cp1252' in console_encoding:
            return None
        return None

    def Colorize(self, string, color, justify=None):
        """Generates a colorized string, optionally justified.

    Args:
      string: The string to write.
      color: The color name -- must be in _ANSI_COLOR.
      justify: The justification function, no justification if None. For
        example, justify=lambda s: s.center(10)

    Returns:
      str, The colorized string that can be printed to the console.
    """
        if justify:
            string = justify(string)
        if self._csi and color in self._ANSI_COLOR:
            return '{csi}{color_code}{string}{csi}{reset_code}'.format(csi=self._csi, color_code=self._ANSI_COLOR[color], reset_code=self._ANSI_COLOR_RESET, string=string)
        return string

    def ConvertOutputToUnicode(self, buf):
        """Converts a console output string buf to unicode.

    Mainly used for testing. Allows test comparisons in unicode while ensuring
    that unicode => encoding => unicode works.

    Args:
      buf: The console output string to convert.

    Returns:
      The console output string buf converted to unicode.
    """
        if isinstance(buf, six.text_type):
            buf = buf.encode(self._encoding)
        return six.text_type(buf, self._encoding, 'replace')

    def GetBoxLineCharacters(self):
        """Returns the box/line drawing characters object.

    The element names are from ISO 8879:1986//ENTITIES Box and Line Drawing//EN:
      http://www.w3.org/2003/entities/iso8879doc/isobox.html

    Returns:
      A BoxLineCharacters object for the console output device.
    """
        return self._box_line_characters

    def GetBullets(self):
        """Returns the bullet characters list.

    Use the list elements in order for best appearance in nested bullet lists,
    wrapping back to the first element for deep nesting. The list size depends
    on the console implementation.

    Returns:
      A tuple of bullet characters.
    """
        return self._bullets

    def GetProgressTrackerSymbols(self):
        """Returns the progress tracker characters object.

    Returns:
      A ProgressTrackerSymbols object for the console output device.
    """
        return self._progress_tracker_symbols

    def GetControlSequenceIndicator(self):
        """Returns the control sequence indicator string.

    Returns:
      The conrol sequence indicator string or None if control sequences are not
      supported.
    """
        return self._csi

    def GetControlSequenceLen(self, buf):
        """Returns the control sequence length at the beginning of buf.

    Used in display width computations. Control sequences have display width 0.

    Args:
      buf: The string to check for a control sequence.

    Returns:
      The conrol sequence length at the beginning of buf or 0 if buf does not
      start with a control sequence.
    """
        if not self._csi or not buf.startswith(self._csi):
            return 0
        n = 0
        for c in buf:
            n += 1
            if c.isalpha():
                break
        return n

    def GetEncoding(self):
        """Returns the current encoding."""
        return self._encoding

    def GetFontCode(self, bold=False, italic=False):
        """Returns a font code string for 0 or more embellishments.

    GetFontCode() with no args returns the default font code string.

    Args:
      bold: True for bold embellishment.
      italic: True for italic embellishment.

    Returns:
      The font code string for the requested embellishments. Write this string
        to the console output to control the font settings.
    """
        if not self._csi:
            return ''
        codes = []
        if bold:
            codes.append(self._font_bold)
        if italic:
            codes.append(self._font_italic)
        return '{csi}{codes}m'.format(csi=self._csi, codes=';'.join(codes))

    def Emphasize(self, s, bold=True, italic=False):
        """Returns a string emphasized."""
        if self._csi:
            s = s.replace(self._csi + self._ANSI_COLOR_RESET, self._csi + self._ANSI_COLOR_RESET + self.GetFontCode(bold, italic))
        return ('{start}' + s + '{end}').format(start=self.GetFontCode(bold, italic), end=self.GetFontCode())

    def GetRawKey(self):
        """Reads one key press from stdin with no echo.

    Returns:
      The key name, None for EOF, <KEY-*> for function keys, otherwise a
      character.
    """
        return self._get_raw_key[0]()

    def GetTermIdentifier(self):
        """Returns the TERM envrionment variable for the console.

    Returns:
      str: A str that describes the console's text capabilities
    """
        return self._term

    def GetTermSize(self):
        """Returns the terminal (x, y) dimensions in characters.

    Returns:
      (x, y): A tuple of the terminal x and y dimensions.
    """
        return self._term_size

    def DisplayWidth(self, buf):
        """Returns the display width of buf, handling unicode and ANSI controls.

    Args:
      buf: The string to count from.

    Returns:
      The display width of buf, handling unicode and ANSI controls.
    """
        if not isinstance(buf, six.string_types):
            return len(buf)
        cached = self._display_width_cache.get(buf, None)
        if cached is not None:
            return cached
        width = 0
        max_width = 0
        i = 0
        while i < len(buf):
            if self._csi and buf[i:].startswith(self._csi):
                i += self.GetControlSequenceLen(buf[i:])
            elif buf[i] == '\n':
                max_width = max(width, max_width)
                width = 0
                i += 1
            else:
                width += GetCharacterDisplayWidth(buf[i])
                i += 1
        max_width = max(width, max_width)
        self._display_width_cache[buf] = max_width
        return max_width

    def SplitIntoNormalAndControl(self, buf):
        """Returns a list of (normal_string, control_sequence) tuples from buf.

    Args:
      buf: The input string containing one or more control sequences
        interspersed with normal strings.

    Returns:
      A list of (normal_string, control_sequence) tuples.
    """
        if not self._csi or not buf:
            return [(buf, '')]
        seq = []
        i = 0
        while i < len(buf):
            c = buf.find(self._csi, i)
            if c < 0:
                seq.append((buf[i:], ''))
                break
            normal = buf[i:c]
            i = c + self.GetControlSequenceLen(buf[c:])
            seq.append((normal, buf[c:i]))
        return seq

    def SplitLine(self, line, width):
        """Splits line into width length chunks.

    Args:
      line: The line to split.
      width: The width of each chunk except the last which could be smaller than
        width.

    Returns:
      A list of chunks, all but the last with display width == width.
    """
        lines = []
        chunk = ''
        w = 0
        keep = False
        for normal, control in self.SplitIntoNormalAndControl(line):
            keep = True
            while True:
                n = width - w
                w += len(normal)
                if w <= width:
                    break
                lines.append(chunk + normal[:n])
                chunk = ''
                keep = False
                w = 0
                normal = normal[n:]
            chunk += normal + control
        if chunk or keep:
            lines.append(chunk)
        return lines

    def SupportsAnsi(self):
        """Indicates whether the terminal appears to support ANSI escape sequences.

    Returns:
      bool: True if ANSI seems to be supported; False otherwise.
    """
        if console_attr_os.ForceEnableAnsi():
            return True
        return self._encoding != 'ascii' and ('screen' in self._term or 'xterm' in self._term)