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
@pytest.mark.parametrize('text', ['lorem'])
def test_tokenizer_handles_single_word(tokenizer, text):
    tokens = tokenizer(text)
    assert tokens[0].text == text