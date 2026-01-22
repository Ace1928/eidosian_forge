import collections
import gzip
import io
import logging
import struct
import numpy as np
def _args_save(fout, model, fb_fasttext_parameters):
    """
    Saves header with `model` parameters to the binary stream `fout` containing a model in the Facebook's
    native fastText `.bin` format.

    Name mimics original C++ implementation, see
    [Args::save](https://github.com/facebookresearch/fastText/blob/master/src/args.cc)

    Parameters
    ----------
    fout: writeable binary stream
        stream to which model is saved
    model: gensim.models.fasttext.FastText
        saved model
    fb_fasttext_parameters: dictionary
        dictionary contain parameters containing `lr_update_rate`, `word_ngrams`
        unused by gensim implementation, so they have to be provided externally
    """
    for field, field_type in _NEW_HEADER_FORMAT:
        if field in fb_fasttext_parameters:
            field_value = fb_fasttext_parameters[field]
        else:
            field_value = _get_field_from_model(model, field)
        fout.write(_conv_field_to_bytes(field_value, field_type))