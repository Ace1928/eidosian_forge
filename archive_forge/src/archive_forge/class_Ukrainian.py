from typing import Callable, Optional
from thinc.api import Model
from ...language import BaseDefaults, Language
from ..punctuation import (
from .lemmatizer import UkrainianLemmatizer
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
class Ukrainian(Language):
    lang = 'uk'
    Defaults = UkrainianDefaults