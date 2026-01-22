import collections
import gzip
import io
import logging
import struct
import numpy as np
def _input_save(fout, model):
    """
    Saves word and ngram vectors from `model` to the binary stream `fout` containing a model in
    the Facebook's native fastText `.bin` format.

    Corresponding C++ fastText code:
    [DenseMatrix::save](https://github.com/facebookresearch/fastText/blob/master/src/densematrix.cc)

    Parameters
    ----------
    fout: writeable binary stream
        stream to which the vectors are saved
    model: gensim.models.fasttext.FastText
        the model that contains the vectors to save
    """
    vocab_n, vocab_dim = model.wv.vectors_vocab.shape
    ngrams_n, ngrams_dim = model.wv.vectors_ngrams.shape
    assert vocab_dim == ngrams_dim
    assert vocab_n == len(model.wv)
    assert ngrams_n == model.wv.bucket
    fout.write(struct.pack('@2q', vocab_n + ngrams_n, vocab_dim))
    fout.write(model.wv.vectors_vocab.tobytes())
    fout.write(model.wv.vectors_ngrams.tobytes())