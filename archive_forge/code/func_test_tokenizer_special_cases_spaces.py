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
def test_tokenizer_special_cases_spaces(tokenizer):
    assert [t.text for t in tokenizer('a b c')] == ['a', 'b', 'c']
    tokenizer.add_special_case('a b c', [{'ORTH': 'a b c'}])
    assert [t.text for t in tokenizer('a b c')] == ['a b c']