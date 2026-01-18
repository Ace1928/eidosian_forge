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
@pytest.mark.parametrize('file_name', ['sun.txt'])
def test_tokenizer_handle_text_from_file(tokenizer, file_name):
    loc = ensure_path(__file__).parent / file_name
    with loc.open('r', encoding='utf8') as infile:
        text = infile.read()
    assert len(text) != 0
    tokens = tokenizer(text)
    assert len(tokens) > 100