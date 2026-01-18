import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_tokens_sent(doc):
    """Test token.sent property"""
    assert len(list(doc.sents)) == 3
    assert doc[1].sent.text == 'This is a sentence .'
    assert doc[7].sent.text == 'This is another sentence .'
    assert doc[1].sent.root.left_edge.text == 'This'
    assert doc[7].sent.root.left_edge.text == 'This'