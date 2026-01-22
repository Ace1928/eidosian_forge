import unicodedata
from wcwidth import wcwidth
from IPython.core.completer import (
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.patch_stdout import patch_stdout
import pygments.lexers as pygments_lexers
import os
import sys
import traceback
class IPythonPTLexer(Lexer):
    """
    Wrapper around PythonLexer and BashLexer.
    """

    def __init__(self):
        l = pygments_lexers
        self.python_lexer = PygmentsLexer(l.Python3Lexer)
        self.shell_lexer = PygmentsLexer(l.BashLexer)
        self.magic_lexers = {'HTML': PygmentsLexer(l.HtmlLexer), 'html': PygmentsLexer(l.HtmlLexer), 'javascript': PygmentsLexer(l.JavascriptLexer), 'js': PygmentsLexer(l.JavascriptLexer), 'perl': PygmentsLexer(l.PerlLexer), 'ruby': PygmentsLexer(l.RubyLexer), 'latex': PygmentsLexer(l.TexLexer)}

    def lex_document(self, document):
        text = document.text.lstrip()
        lexer = self.python_lexer
        if text.startswith('!') or text.startswith('%%bash'):
            lexer = self.shell_lexer
        elif text.startswith('%%'):
            for magic, l in self.magic_lexers.items():
                if text.startswith('%%' + magic):
                    lexer = l
                    break
        return lexer.lex_document(document)