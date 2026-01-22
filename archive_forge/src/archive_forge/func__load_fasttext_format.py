import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
def _load_fasttext_format(model_file, encoding='utf-8', full_model=True):
    """Load the input-hidden weight matrix from Facebook's native fasttext `.bin` output files.

    Parameters
    ----------
    model_file : str
        Full path to the FastText model file.
    encoding : str, optional
        Specifies the file encoding.
    full_model : boolean, optional
        If False, skips loading the hidden output matrix. This saves a fair bit
        of CPU time and RAM, but prevents training continuation.

    Returns
    -------
    :class: `~gensim.models.fasttext.FastText`
        The loaded model.

    """
    with utils.open(model_file, 'rb') as fin:
        m = gensim.models._fasttext_bin.load(fin, encoding=encoding, full_model=full_model)
    model = FastText(vector_size=m.dim, window=m.ws, epochs=m.epoch, negative=m.neg, hs=int(m.loss == 1), sg=int(m.model == 2), bucket=m.bucket, min_count=m.min_count, sample=m.t, min_n=m.minn, max_n=m.maxn)
    model.corpus_total_words = m.ntokens
    model.raw_vocab = m.raw_vocab
    model.nwords = m.nwords
    model.vocab_size = m.vocab_size
    model.prepare_vocab(update=True, min_count=1)
    model.num_original_vectors = m.vectors_ngrams.shape[0]
    model.wv.init_post_load(m.vectors_ngrams)
    model._init_post_load(m.hidden_output)
    _check_model(model)
    model.add_lifecycle_event('load_fasttext_format', msg=f'loaded {m.vectors_ngrams.shape} weight matrix for fastText model from {fin.name}')
    return model