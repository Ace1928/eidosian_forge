import pytest
from spacy.lang.char_classes import ALPHA
from spacy.lang.punctuation import TOKENIZER_INFIXES
from spacy.language import BaseDefaults, Language
class FrenchTest(Language):

    class Defaults(BaseDefaults):
        infixes = TOKENIZER_INFIXES + [SPLIT_INFIX]