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
@pytest.mark.parametrize('text,tokens', [('lorem', [{'orth': 'lo'}, {'orth': 'rem'}])])
def test_tokenizer_add_special_case(tokenizer, text, tokens):
    tokenizer.add_special_case(text, tokens)
    doc = tokenizer(text)
    assert doc[0].text == tokens[0]['orth']
    assert doc[1].text == tokens[1]['orth']