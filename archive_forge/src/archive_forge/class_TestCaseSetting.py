import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class TestCaseSetting(Setting):
    _keyword_settings = ('setup', 'precondition', 'teardown', 'postcondition', 'template')
    _import_settings = ()
    _other_settings = ('documentation', 'tags', 'timeout')

    def _tokenize(self, value, index):
        if index == 0:
            type = Setting._tokenize(self, value[1:-1], index)
            return [('[', SYNTAX), (value[1:-1], type), (']', SYNTAX)]
        return Setting._tokenize(self, value, index)