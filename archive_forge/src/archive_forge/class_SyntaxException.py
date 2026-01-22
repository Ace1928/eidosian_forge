import sys
import traceback
from mako import compat
from mako import util
class SyntaxException(MakoException):

    def __init__(self, message, source, lineno, pos, filename):
        MakoException.__init__(self, message + _format_filepos(lineno, pos, filename))
        self.lineno = lineno
        self.pos = pos
        self.filename = filename
        self.source = source