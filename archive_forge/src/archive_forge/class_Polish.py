from typing import Callable, Optional
from thinc.api import Model
from ...language import BaseDefaults, Language
from ..tokenizer_exceptions import BASE_EXCEPTIONS
from .lemmatizer import PolishLemmatizer
from .lex_attrs import LEX_ATTRS
from .punctuation import TOKENIZER_INFIXES, TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES
from .stop_words import STOP_WORDS
class Polish(Language):
    lang = 'pl'
    Defaults = PolishDefaults