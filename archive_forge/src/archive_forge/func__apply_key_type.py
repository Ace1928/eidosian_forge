from itertools import cycle
from operator import itemgetter
import numpy as np
import pandas as pd
import param
from . import util
from .dimension import Dimension, Dimensioned, ViewableElement, asdim
from .util import (
def _apply_key_type(self, keys):
    """
        If a type is specified by the corresponding key dimension,
        this method applies the type to the supplied key.
        """
    typed_key = ()
    for dim, key in zip(self.kdims, keys):
        key_type = dim.type
        if key_type is None:
            typed_key += (key,)
        elif isinstance(key, slice):
            sl_vals = [key.start, key.stop, key.step]
            typed_key += (slice(*[key_type(el) if el is not None else None for el in sl_vals]),)
        elif key is Ellipsis:
            typed_key += (key,)
        elif isinstance(key, list):
            typed_key += ([key_type(k) for k in key],)
        else:
            typed_key += (key_type(key),)
    return typed_key