import warnings
import pytest
import srsly
from mock import Mock
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_phrase_matcher_validation(en_vocab):
    doc1 = Doc(en_vocab, words=['Test'])
    doc1[0].dep_ = 'ROOT'
    doc2 = Doc(en_vocab, words=['Test'])
    doc2[0].tag_ = 'TAG'
    doc2[0].pos_ = 'X'
    doc2[0].set_morph('Feat=Val')
    doc3 = Doc(en_vocab, words=['Test'])
    matcher = PhraseMatcher(en_vocab, validate=True)
    with pytest.warns(UserWarning):
        matcher.add('TEST1', [doc1])
    with pytest.warns(UserWarning):
        matcher.add('TEST2', [doc2])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        matcher.add('TEST3', [doc3])
    matcher = PhraseMatcher(en_vocab, attr='POS', validate=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        matcher.add('TEST4', [doc2])