from typing import Callable, Optional
from thinc.api import Model
from ...language import BaseDefaults, Language
from .lemmatizer import DutchLemmatizer
from .lex_attrs import LEX_ATTRS
from .punctuation import TOKENIZER_INFIXES, TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES
from .stop_words import STOP_WORDS
from .syntax_iterators import SYNTAX_ITERATORS
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
class Dutch(Language):
    lang = 'nl'
    Defaults = DutchDefaults