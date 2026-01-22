import numpy as np
from ._miobase import convert_dtypes
def _convert_codecs(template, byte_order):
    """ Convert codec template mapping to byte order

    Set codecs not on this system to None

    Parameters
    ----------
    template : mapping
       key, value are respectively codec name, and root name for codec
       (without byte order suffix)
    byte_order : {'<', '>'}
       code for little or big endian

    Returns
    -------
    codecs : dict
       key, value are name, codec (as in .encode(codec))
    """
    codecs = {}
    postfix = byte_order == '<' and '_le' or '_be'
    for k, v in template.items():
        codec = v['codec']
        try:
            ' '.encode(codec)
        except LookupError:
            codecs[k] = None
            continue
        if v['width'] > 1:
            codec += postfix
        codecs[k] = codec
    return codecs.copy()