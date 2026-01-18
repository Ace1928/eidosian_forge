import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_token0_has_sent_start_true():
    doc = Doc(Vocab(), words=['hello', 'world'])
    assert doc[0].is_sent_start is True
    assert doc[1].is_sent_start is None
    assert not doc.has_annotation('SENT_START')