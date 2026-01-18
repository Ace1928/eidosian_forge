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
def test_tokenizer_special_cases_with_affixes(tokenizer):
    text = '(((_SPECIAL_ A/B, A/B-A/B")'
    tokenizer.add_special_case('_SPECIAL_', [{'orth': '_SPECIAL_'}])
    tokenizer.add_special_case('A/B', [{'orth': 'A/B'}])
    doc = tokenizer(text)
    assert [token.text for token in doc] == ['(', '(', '(', '_SPECIAL_', 'A/B', ',', 'A/B', '-', 'A/B', '"', ')']