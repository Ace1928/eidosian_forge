from ...language import BaseDefaults, Language
from .lex_attrs import LEX_ATTRS
from .punctuation import TOKENIZER_SUFFIXES
from .stop_words import STOP_WORDS
class Urdu(Language):
    lang = 'ur'
    Defaults = UrduDefaults