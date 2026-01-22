import re
from pygments.lexer import Lexer
from pygments.token import Token, Error, Text
from pygments.util import get_choice_opt, text_type, BytesIO
class TextLexer(Lexer):
    """
    "Null" lexer, doesn't highlight anything.
    """
    name = 'Text only'
    aliases = ['text']
    filenames = ['*.txt']
    mimetypes = ['text/plain']
    priority = 0.01

    def get_tokens_unprocessed(self, text):
        yield (0, Text, text)

    def analyse_text(text):
        return TextLexer.priority