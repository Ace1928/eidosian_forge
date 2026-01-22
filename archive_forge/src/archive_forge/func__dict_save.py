import collections
import gzip
import io
import logging
import struct
import numpy as np
def _dict_save(fout, model, encoding):
    """
    Saves the dictionary from `model` to the to the binary stream `fout` containing a model in the Facebook's
    native fastText `.bin` format.

    Name mimics the original C++ implementation
    [Dictionary::save](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)

    Parameters
    ----------
    fout: writeable binary stream
        stream to which the dictionary from the model is saved
    model: gensim.models.fasttext.FastText
        the model that contains the dictionary to save
    encoding: str
        string encoding used in the output
    """
    fout.write(np.int32(len(model.wv)).tobytes())
    fout.write(np.int32(len(model.wv)).tobytes())
    fout.write(np.int32(0).tobytes())
    fout.write(np.int64(model.corpus_total_words).tobytes())
    fout.write(np.int64(-1))
    for word in model.wv.index_to_key:
        word_count = model.wv.get_vecattr(word, 'count')
        fout.write(word.encode(encoding))
        fout.write(_END_OF_WORD_MARKER)
        fout.write(np.int64(word_count).tobytes())
        fout.write(_DICT_WORD_ENTRY_TYPE_MARKER)