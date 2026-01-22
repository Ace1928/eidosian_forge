from pygments.formatter import Formatter
from pygments.util import OptionError, get_choice_opt
from pygments.token import Token
from pygments.console import colorize
class NullFormatter(Formatter):
    """
    Output the text unchanged without any formatting.
    """
    name = 'Text only'
    aliases = ['text', 'null']
    filenames = ['*.txt']

    def format(self, tokensource, outfile):
        enc = self.encoding
        for ttype, value in tokensource:
            if enc:
                outfile.write(value.encode(enc))
            else:
                outfile.write(value)