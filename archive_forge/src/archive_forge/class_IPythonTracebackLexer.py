import re
from pygments.lexers import (
from pygments.lexer import (
from pygments.token import (
from pygments.util import get_bool_opt
class IPythonTracebackLexer(DelegatingLexer):
    """
    IPython traceback lexer.

    For doctests, the tracebacks can be snipped as much as desired with the
    exception to the lines that designate a traceback. For non-syntax error
    tracebacks, this is the line of hyphens. For syntax error tracebacks,
    this is the line which lists the File and line number.

    """
    name = 'IPython Traceback'
    aliases = ['ipythontb']

    def __init__(self, **options):
        """
        A subclass of `DelegatingLexer` which delegates to the appropriate to either IPyLexer,
        IPythonPartialTracebackLexer.
        """
        self.python3 = get_bool_opt(options, 'python3', False)
        if self.python3:
            self.aliases = ['ipython3tb']
        else:
            self.aliases = ['ipython2tb', 'ipythontb']
        if self.python3:
            IPyLexer = IPython3Lexer
        else:
            IPyLexer = IPythonLexer
        DelegatingLexer.__init__(self, IPyLexer, IPythonPartialTracebackLexer, **options)