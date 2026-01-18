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
def test_tokenizer_handles_digits(tokenizer):
    exceptions = ['hu', 'bn']
    text = 'Lorem ipsum: 1984.'
    tokens = tokenizer(text)
    if tokens[0].lang_ not in exceptions:
        assert len(tokens) == 5
        assert tokens[0].text == 'Lorem'
        assert tokens[3].text == '1984'