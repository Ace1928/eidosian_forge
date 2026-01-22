from ...attrs import LANG
from ...language import BaseDefaults, Language
from ...util import update_exc
from ..punctuation import (
from ..tokenizer_exceptions import BASE_EXCEPTIONS
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
class Bulgarian(Language):
    lang = 'bg'
    Defaults = BulgarianDefaults