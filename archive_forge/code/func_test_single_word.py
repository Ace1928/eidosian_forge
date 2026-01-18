import pytest
from spacy import util
from spacy.tokens import Doc
from spacy.vocab import Vocab
def test_single_word(vocab):
    doc = Doc(vocab, words=['a'])
    assert doc.text == 'a '
    doc = Doc(vocab, words=['a'], spaces=[False])
    assert doc.text == 'a'