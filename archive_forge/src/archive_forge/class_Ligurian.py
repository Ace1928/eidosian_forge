from ...language import BaseDefaults, Language
from .punctuation import TOKENIZER_INFIXES
from .stop_words import STOP_WORDS
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
class Ligurian(Language):
    lang = 'lij'
    Defaults = LigurianDefaults