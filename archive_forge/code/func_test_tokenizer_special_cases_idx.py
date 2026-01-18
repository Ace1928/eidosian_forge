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
def test_tokenizer_special_cases_idx(tokenizer):
    text = "the _ID'X_"
    tokenizer.add_special_case("_ID'X_", [{'orth': '_ID'}, {'orth': "'X_"}])
    doc = tokenizer(text)
    assert doc[1].idx == 4
    assert doc[2].idx == 7