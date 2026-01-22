from __future__ import absolute_import
import pickle as _pickle
from smart_open import open
from gensim import utils
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors
def _build_from_word2vec(self):
    """Build an NMSLIB index using word vectors from a Word2Vec model."""
    self._build_from_model(self.model.wv.get_normed_vectors(), self.model.wv.index_to_key)