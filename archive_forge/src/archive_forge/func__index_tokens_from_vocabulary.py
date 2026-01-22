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
def _index_tokens_from_vocabulary(self, vocabulary):
    self._token_to_idx = vocabulary.token_to_idx.copy() if vocabulary.token_to_idx is not None else None
    self._idx_to_token = vocabulary.idx_to_token[:] if vocabulary.idx_to_token is not None else None
    self._unknown_token = vocabulary.unknown_token
    self._reserved_tokens = vocabulary.reserved_tokens[:] if vocabulary.reserved_tokens is not None else None