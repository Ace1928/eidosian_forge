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
def _set_idx_to_vec_by_embeddings(self, token_embeddings, vocab_len, vocab_idx_to_token):
    """Sets the mapping between token indices and token embedding vectors.


        Parameters
        ----------
        token_embeddings : instance or list `mxnet.contrib.text.embedding._TokenEmbedding`
            One or multiple pre-trained token embeddings to load. If it is a list of multiple
            embeddings, these embedding vectors will be concatenated for each token.
        vocab_len : int
            Length of vocabulary whose tokens are indexed in the token embedding.
        vocab_idx_to_token: list of str
            A list of indexed tokens in the vocabulary. These tokens are indexed in the token
            embedding.
        """
    new_vec_len = sum((embed.vec_len for embed in token_embeddings))
    zeros_fn = _mx_np.zeros if is_np_array() else nd.zeros
    new_idx_to_vec = zeros_fn(shape=(vocab_len, new_vec_len))
    col_start = 0
    for embed in token_embeddings:
        col_end = col_start + embed.vec_len
        new_idx_to_vec[0, col_start:col_end] = embed.idx_to_vec[0]
        new_idx_to_vec[1:, col_start:col_end] = embed.get_vecs_by_tokens(vocab_idx_to_token[1:])
        col_start = col_end
    self._vec_len = new_vec_len
    self._idx_to_vec = new_idx_to_vec