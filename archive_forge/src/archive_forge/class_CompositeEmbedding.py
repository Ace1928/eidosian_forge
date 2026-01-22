import logging
import os
import tarfile
import warnings
import zipfile
from . import _constants as C
from . import vocab
from ... import ndarray as nd
from ... import registry
from ... import base
from ...util import is_np_array
from ... import numpy as _mx_np
from ... import numpy_extension as _mx_npx
class CompositeEmbedding(_TokenEmbedding):
    """Composite token embeddings.


    For each indexed token in a vocabulary, multiple embedding vectors, such as concatenated
    multiple embedding vectors, will be associated with it. Such embedding vectors can be loaded
    from externally hosted or custom pre-trained token embedding files, such as via token embedding
    instances.


    Parameters
    ----------
    vocabulary : :class:`~mxnet.contrib.text.vocab.Vocabulary`
        For each indexed token in a vocabulary, multiple embedding vectors, such as concatenated
        multiple embedding vectors, will be associated with it.
    token_embeddings : instance or list of `mxnet.contrib.text.embedding._TokenEmbedding`
        One or multiple pre-trained token embeddings to load. If it is a list of multiple
        embeddings, these embedding vectors will be concatenated for each token.
    """

    def __init__(self, vocabulary, token_embeddings):
        assert isinstance(vocabulary, vocab.Vocabulary), 'The argument `vocabulary` must be an instance of mxnet.contrib.text.indexer.Vocabulary.'
        if not isinstance(token_embeddings, list):
            token_embeddings = [token_embeddings]
        for embed in token_embeddings:
            assert isinstance(embed, _TokenEmbedding), 'The argument `token_embeddings` must be an instance or a list of instances of `mxnet.contrib.text.embedding.TextEmbedding` whose embedding vectors will beloaded or concatenated-then-loaded to map to the indexed tokens.'
        self._index_tokens_from_vocabulary(vocabulary)
        self._set_idx_to_vec_by_embeddings(token_embeddings, len(self), self.idx_to_token)