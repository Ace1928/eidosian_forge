from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
@staticmethod
def _validate_and_normalize_without_numpy(data, normalize):
    import array
    import sys
    data = array.array('f', data)
    try:
        max_abs_value = float(max([abs(x) for x in data]))
    except TypeError as e:
        raise TypeError('Only lists of mono audio are supported if numpy is not installed') from e
    normalization_factor = Audio._get_normalization_factor(max_abs_value, normalize)
    scaled = array.array('h', [int(x / normalization_factor * 32767) for x in data])
    if sys.byteorder == 'big':
        scaled.byteswap()
    nchan = 1
    return (scaled.tobytes(), nchan)