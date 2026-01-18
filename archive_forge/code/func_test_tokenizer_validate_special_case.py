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
@pytest.mark.parametrize('text,tokens', [('lorem', [{'orth': 'lo'}, {'orth': 're'}]), ('lorem', [{'orth': 'lo', 'tag': 'A'}, {'orth': 'rem'}])])
def test_tokenizer_validate_special_case(tokenizer, text, tokens):
    with pytest.raises(ValueError):
        tokenizer.add_special_case(text, tokens)