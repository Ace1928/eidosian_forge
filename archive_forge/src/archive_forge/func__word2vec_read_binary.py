import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import (
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
def _word2vec_read_binary(fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, binary_chunk_size, encoding='utf-8'):
    chunk = b''
    tot_processed_words = 0
    while tot_processed_words < vocab_size:
        new_chunk = fin.read(binary_chunk_size)
        chunk += new_chunk
        processed_words, chunk = _add_bytes_to_kv(kv, counts, chunk, vocab_size, vector_size, datatype, unicode_errors, encoding)
        tot_processed_words += processed_words
        if len(new_chunk) < binary_chunk_size:
            break
    if tot_processed_words != vocab_size:
        raise EOFError('unexpected end of input; is count incorrect or file otherwise damaged?')