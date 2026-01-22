import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class Setting(Tokenizer):
    _tokens = (SETTING, ARGUMENT)
    _keyword_settings = ('suitesetup', 'suiteprecondition', 'suiteteardown', 'suitepostcondition', 'testsetup', 'testprecondition', 'testteardown', 'testpostcondition', 'testtemplate')
    _import_settings = ('library', 'resource', 'variables')
    _other_settings = ('documentation', 'metadata', 'forcetags', 'defaulttags', 'testtimeout')
    _custom_tokenizer = None

    def __init__(self, template_setter=None):
        Tokenizer.__init__(self)
        self._template_setter = template_setter

    def _tokenize(self, value, index):
        if index == 1 and self._template_setter:
            self._template_setter(value)
        if index == 0:
            normalized = normalize(value)
            if normalized in self._keyword_settings:
                self._custom_tokenizer = KeywordCall(support_assign=False)
            elif normalized in self._import_settings:
                self._custom_tokenizer = ImportSetting()
            elif normalized not in self._other_settings:
                return ERROR
        elif self._custom_tokenizer:
            return self._custom_tokenizer.tokenize(value)
        return Tokenizer._tokenize(self, value, index)