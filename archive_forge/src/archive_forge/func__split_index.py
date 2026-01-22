from itertools import cycle
from operator import itemgetter
import numpy as np
import pandas as pd
import param
from . import util
from .dimension import Dimension, Dimensioned, ViewableElement, asdim
from .util import (
def _split_index(self, key):
    """
        Partitions key into key and deep dimension groups. If only key
        indices are supplied, the data is indexed with an empty tuple.
        Keys with indices than there are dimensions will be padded.
        """
    if not isinstance(key, tuple):
        key = (key,)
    elif key == ():
        return ((), ())
    if key[0] is Ellipsis:
        num_pad = self.ndims - len(key) + 1
        key = (slice(None),) * num_pad + key[1:]
    elif len(key) < self.ndims:
        num_pad = self.ndims - len(key)
        key = key + (slice(None),) * num_pad
    map_slice = key[:self.ndims]
    if self._check_key_type:
        map_slice = self._apply_key_type(map_slice)
    if len(key) == self.ndims:
        return (map_slice, ())
    else:
        return (map_slice, key[self.ndims:])