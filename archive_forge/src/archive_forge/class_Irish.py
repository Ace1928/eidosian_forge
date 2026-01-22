from typing import Optional
from thinc.api import Model
from ...language import BaseDefaults, Language
from .lemmatizer import IrishLemmatizer
from .stop_words import STOP_WORDS
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
class Irish(Language):
    lang = 'ga'
    Defaults = IrishDefaults