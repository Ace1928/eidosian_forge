import collections
import gzip
import io
import logging
import struct
import numpy as np

    Saves word embeddings to the Facebook's native fasttext `.bin` format.

    Parameters
    ----------
    fout: file name or writeable binary stream
        stream to which model is saved
    model: gensim.models.fasttext.FastText
        saved model
    fb_fasttext_parameters: dictionary
        dictionary contain parameters containing `lr_update_rate`, `word_ngrams`
        unused by gensim implementation, so they have to be provided externally
    encoding: str
        encoding used in the output file

    Notes
    -----
    Unfortunately, there is no documentation of the Facebook's native fasttext `.bin` format

    This is just reimplementation of
    [FastText::saveModel](https://github.com/facebookresearch/fastText/blob/master/src/fasttext.cc)

    Based on v0.9.1, more precisely commit da2745fcccb848c7a225a7d558218ee4c64d5333

    Code follows the original C++ code naming.
    