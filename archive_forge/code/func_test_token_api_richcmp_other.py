import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_token_api_richcmp_other(en_tokenizer):
    doc1 = en_tokenizer('a b')
    doc2 = en_tokenizer('b c')
    assert not doc1[1] == doc1[0:1]
    assert not doc1[1] == doc2[1:2]
    assert not doc1[1] == doc2[0]
    assert not doc1[0] == doc2