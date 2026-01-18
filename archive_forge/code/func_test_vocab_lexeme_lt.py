import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT
from spacy.lookups import Lookups
from spacy.tokens import Doc
from spacy.util import OOV_RANK
from spacy.vocab import Vocab
@pytest.mark.parametrize('text1,prob1,text2,prob2', [('NOUN', -1, 'opera', -2)])
def test_vocab_lexeme_lt(en_vocab, text1, text2, prob1, prob2):
    """More frequent is l.t. less frequent"""
    lex1 = en_vocab[text1]
    lex1.prob = prob1
    lex2 = en_vocab[text2]
    lex2.prob = prob2
    assert lex1 < lex2
    assert lex2 > lex1