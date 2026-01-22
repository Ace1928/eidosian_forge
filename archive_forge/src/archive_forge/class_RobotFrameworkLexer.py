import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class RobotFrameworkLexer(Lexer):
    """
    For `Robot Framework <http://robotframework.org>`_ test data.

    Supports both space and pipe separated plain text formats.

    .. versionadded:: 1.6
    """
    name = 'RobotFramework'
    aliases = ['robotframework']
    filenames = ['*.txt', '*.robot']
    mimetypes = ['text/x-robotframework']

    def __init__(self, **options):
        options['tabsize'] = 2
        options['encoding'] = 'UTF-8'
        Lexer.__init__(self, **options)

    def get_tokens_unprocessed(self, text):
        row_tokenizer = RowTokenizer()
        var_tokenizer = VariableTokenizer()
        index = 0
        for row in text.splitlines():
            for value, token in row_tokenizer.tokenize(row):
                for value, token in var_tokenizer.tokenize(value, token):
                    if value:
                        yield (index, token, text_type(value))
                        index += len(value)