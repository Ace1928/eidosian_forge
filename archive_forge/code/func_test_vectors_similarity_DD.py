import numpy
import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab, get_cosine
def test_vectors_similarity_DD(vocab, vectors):
    [(word1, vec1), (word2, vec2)] = vectors
    doc1 = Doc(vocab, words=[word1, word2])
    doc2 = Doc(vocab, words=[word2, word1])
    assert isinstance(doc1.similarity(doc2), float)
    assert doc1.similarity(doc2) == doc2.similarity(doc1)