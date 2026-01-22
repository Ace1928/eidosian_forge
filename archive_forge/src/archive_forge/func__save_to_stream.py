import collections
import gzip
import io
import logging
import struct
import numpy as np
def _save_to_stream(model, fout, fb_fasttext_parameters, encoding):
    """
    Saves word embeddings to binary stream `fout` using the Facebook's native fasttext `.bin` format.

    Parameters
    ----------
    fout: file name or writeable binary stream
        stream to which the word embeddings are saved
    model: gensim.models.fasttext.FastText
        the model that contains the word embeddings to save
    fb_fasttext_parameters: dictionary
        dictionary contain parameters containing `lr_update_rate`, `word_ngrams`
        unused by gensim implementation, so they have to be provided externally
    encoding: str
        encoding used in the output file
    """
    _sign_model(fout)
    _args_save(fout, model, fb_fasttext_parameters)
    _dict_save(fout, model, encoding)
    fout.write(struct.pack('@?', False))
    _input_save(fout, model)
    fout.write(struct.pack('@?', False))
    _output_save(fout, model)