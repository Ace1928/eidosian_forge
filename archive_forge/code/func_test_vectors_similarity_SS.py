import numpy
import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab, get_cosine
def test_vectors_similarity_SS(vocab, vectors):
    [(word1, vec1), (word2, vec2)] = vectors
    doc = Doc(vocab, words=[word1, word2])
    assert isinstance(doc[0:1].similarity(doc[0:2]), float)
    assert doc[0:1].similarity(doc[0:2]) == doc[0:2].similarity(doc[0:1])