import collections
import gzip
import io
import logging
import struct
import numpy as np
def _get_field_from_model(model, field):
    """
    Extract `field` from `model`.

    Parameters
    ----------
    model: gensim.models.fasttext.FastText
        model from which `field` is extracted
    field: str
        requested field name, fields are listed in the `_NEW_HEADER_FORMAT` list
    """
    if field == 'bucket':
        return model.wv.bucket
    elif field == 'dim':
        return model.vector_size
    elif field == 'epoch':
        return model.epochs
    elif field == 'loss':
        if model.hs == 1:
            return 1
        elif model.hs == 0:
            return 2
        elif model.hs == 0 and model.negative == 0:
            return 1
    elif field == 'maxn':
        return model.wv.max_n
    elif field == 'minn':
        return model.wv.min_n
    elif field == 'min_count':
        return model.min_count
    elif field == 'model':
        return 2 if model.sg == 1 else 1
    elif field == 'neg':
        return model.negative
    elif field == 't':
        return model.sample
    elif field == 'word_ngrams':
        return 1
    elif field == 'ws':
        return model.window
    elif field == 'lr_update_rate':
        return 100
    else:
        msg = 'Extraction of header field "' + field + '" from Gensim FastText object not implemmented.'
        raise NotImplementedError(msg)