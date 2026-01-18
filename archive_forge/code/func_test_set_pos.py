import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_set_pos():
    doc = Doc(Vocab(), words=['hello', 'world'])
    doc[0].pos_ = 'NOUN'
    assert doc[0].pos_ == 'NOUN'
    doc[1].pos = VERB
    assert doc[1].pos_ == 'VERB'