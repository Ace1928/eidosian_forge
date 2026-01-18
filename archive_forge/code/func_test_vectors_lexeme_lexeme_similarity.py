import numpy
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from thinc.api import NumpyOps, get_current_ops
from spacy.lang.en import English
from spacy.strings import hash_string  # type: ignore
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training.initialize import convert_vectors
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab, get_cosine, make_tempdir
@pytest.mark.parametrize('text1,text2', [('apple', 'orange')])
def test_vectors_lexeme_lexeme_similarity(vocab, text1, text2):
    lex1 = vocab[text1]
    lex2 = vocab[text2]
    assert lex1.similarity(lex2) == lex2.similarity(lex1)
    assert -1.0 < lex1.similarity(lex2) < 1.0