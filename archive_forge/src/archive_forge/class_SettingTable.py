import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class SettingTable(_Table):
    _tokenizer_class = Setting

    def __init__(self, template_setter, prev_tokenizer=None):
        _Table.__init__(self, prev_tokenizer)
        self._template_setter = template_setter

    def _tokenize(self, value, index):
        if index == 0 and normalize(value) == 'testtemplate':
            self._tokenizer = Setting(self._template_setter)
        return _Table._tokenize(self, value, index)

    def end_row(self):
        self.__init__(self._template_setter, prev_tokenizer=self._tokenizer)