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
@pytest.mark.parametrize('text1,text2', [(['apple', 'and', 'apple', 'pie'], ['orange', 'juice'])])
def test_vectors_doc_doc_similarity(vocab, text1, text2):
    doc1 = Doc(vocab, words=text1)
    doc2 = Doc(vocab, words=text2)
    assert doc1.similarity(doc2) == doc2.similarity(doc1)
    assert -1.0 < doc1.similarity(doc2) < 1.0