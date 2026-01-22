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
def _execute_callback(self, *args):
    """Executes the callback with the appropriate args and kwargs"""
    self._validate_key(args)
    kdims = [kdim.name for kdim in self.kdims]
    kwarg_items = [s.contents.items() for s in self.streams]
    hash_items = tuple((tuple(sorted(s.hashkey.items())) for s in self.streams)) + args
    flattened = [(k, v) for kws in kwarg_items for k, v in kws if k not in kdims]
    if self.positional_stream_args:
        kwargs = {}
        args = args + tuple([s.contents for s in self.streams])
    elif self._posarg_keys:
        kwargs = dict(flattened, **dict(zip(self._posarg_keys, args)))
        args = ()
    else:
        kwargs = dict(flattened)
    if not isinstance(self.callback, Generator):
        kwargs['_memoization_hash_'] = hash_items
    with dynamicmap_memoization(self.callback, self.streams):
        retval = self.callback(*args, **kwargs)
    return self._style(retval)