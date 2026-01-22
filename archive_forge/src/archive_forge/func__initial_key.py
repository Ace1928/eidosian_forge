import itertools
import types
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import param
from ..streams import Params, Stream, streams_list_from_dict
from . import traversal, util
from .accessors import Opts, Redim
from .dimension import Dimension, ViewableElement
from .layout import AdjointLayout, Empty, Layout, Layoutable, NdLayout
from .ndmapping import NdMapping, UniformNdMapping, item_check
from .options import Store, StoreOptions
from .overlay import CompositeOverlay, NdOverlay, Overlay, Overlayable
def _initial_key(self):
    """
        Construct an initial key for based on the lower range bounds or
        values on the key dimensions.
        """
    key = []
    undefined = []
    stream_params = set(self._stream_parameters())
    for kdim in self.kdims:
        if str(kdim) in stream_params:
            key.append(None)
        elif kdim.default is not None:
            key.append(kdim.default)
        elif kdim.values:
            if all((util.isnumeric(v) for v in kdim.values)):
                key.append(sorted(kdim.values)[0])
            else:
                key.append(kdim.values[0])
        elif kdim.range[0] is not None:
            key.append(kdim.range[0])
        else:
            undefined.append(kdim)
    if undefined:
        msg = 'Dimension(s) {undefined_dims} do not specify range or values needed to generate initial key'
        undefined_dims = ', '.join((f'{str(dim)!r}' for dim in undefined))
        raise KeyError(msg.format(undefined_dims=undefined_dims))
    return tuple(key)