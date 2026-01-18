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
def test_tokenizer_handles_punct(tokenizer):
    text = 'Lorem, ipsum.'
    tokens = tokenizer(text)
    assert len(tokens) == 4
    assert tokens[0].text == 'Lorem'
    assert tokens[1].text == ','
    assert tokens[2].text == 'ipsum'
    assert tokens[1].text != 'Lorem'