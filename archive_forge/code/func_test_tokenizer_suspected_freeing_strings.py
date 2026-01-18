import re
import numpy
import pytest
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import (
from spacy.vocab import Vocab
def test_tokenizer_suspected_freeing_strings(tokenizer):
    text1 = 'Lorem dolor sit amet, consectetur adipiscing elit.'
    text2 = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit.'
    tokens1 = tokenizer(text1)
    tokens2 = tokenizer(text2)
    assert tokens1[0].text == 'Lorem'
    assert tokens2[0].text == 'Lorem'