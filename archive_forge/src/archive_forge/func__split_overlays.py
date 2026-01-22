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
def _split_overlays(self):
    """
        Splits a DynamicMap into its components. Only well defined for
        DynamicMap with consistent number and order of layers.
        """
    if not len(self):
        raise ValueError('Cannot split DynamicMap before it has been initialized')
    elif not issubclass(self.type, CompositeOverlay):
        return (None, self)
    from ..util import Dynamic
    keys = list(self.last.data.keys())
    dmaps = []
    for key in keys:
        el = self.last.data[key]

        def split_overlay_callback(obj, overlay_key=key, overlay_el=el, **kwargs):
            spec = util.get_overlay_spec(obj, overlay_key, overlay_el)
            items = list(obj.data.items())
            specs = [(i, util.get_overlay_spec(obj, k, v)) for i, (k, v) in enumerate(items)]
            match = util.closest_match(spec, specs)
            if match is None:
                otype = type(obj).__name__
                raise KeyError(f'{spec} spec not found in {otype}. The split_overlays method only works consistently for a DynamicMap where the layers of the {otype} do not change.')
            return items[match][1]
        dmap = Dynamic(self, streams=self.streams, operation=split_overlay_callback)
        dmap.data = dict([(list(self.data.keys())[-1], self.last.data[key])])
        dmaps.append(dmap)
    return (keys, dmaps)