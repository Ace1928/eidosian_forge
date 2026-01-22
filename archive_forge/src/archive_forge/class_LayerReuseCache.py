import collections
import copy
import enum
from functools import partial
from math import ceil, log
from typing import (
from fontTools.misc.arrayTools import intRect
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import C_O_L_R_
from fontTools.ttLib.tables import C_P_A_L_
from fontTools.ttLib.tables import _n_a_m_e
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otTables import ExtendMode, CompositeMode
from .errors import ColorLibError
from .geometry import round_start_circle_stable_containment
from .table_builder import BuildCallback, TableBuilder
class LayerReuseCache:
    reusePool: Mapping[Tuple[Any, ...], int]
    tuples: Mapping[int, Tuple[Any, ...]]
    keepAlive: List[ot.Paint]

    def __init__(self):
        self.reusePool = {}
        self.tuples = {}
        self.keepAlive = []

    def _paint_tuple(self, paint: ot.Paint):

        def _tuple_safe(value):
            if isinstance(value, enum.Enum):
                return value
            elif hasattr(value, '__dict__'):
                return tuple(((k, _tuple_safe(v)) for k, v in sorted(value.__dict__.items())))
            elif isinstance(value, collections.abc.MutableSequence):
                return tuple((_tuple_safe(e) for e in value))
            return value
        result = self.tuples.get(id(paint), None)
        if result is None:
            result = _tuple_safe(paint)
            self.tuples[id(paint)] = result
            self.keepAlive.append(paint)
        return result

    def _as_tuple(self, paints: Sequence[ot.Paint]) -> Tuple[Any, ...]:
        return tuple((self._paint_tuple(p) for p in paints))

    def try_reuse(self, layers: List[ot.Paint]) -> List[ot.Paint]:
        found_reuse = True
        while found_reuse:
            found_reuse = False
            ranges = sorted(_reuse_ranges(len(layers)), key=lambda t: (t[1] - t[0], t[1], t[0]), reverse=True)
            for lbound, ubound in ranges:
                reuse_lbound = self.reusePool.get(self._as_tuple(layers[lbound:ubound]), -1)
                if reuse_lbound == -1:
                    continue
                new_slice = ot.Paint()
                new_slice.Format = int(ot.PaintFormat.PaintColrLayers)
                new_slice.NumLayers = ubound - lbound
                new_slice.FirstLayerIndex = reuse_lbound
                layers = layers[:lbound] + [new_slice] + layers[ubound:]
                found_reuse = True
                break
        return layers

    def add(self, layers: List[ot.Paint], first_layer_index: int):
        for lbound, ubound in _reuse_ranges(len(layers)):
            self.reusePool[self._as_tuple(layers[lbound:ubound])] = lbound + first_layer_index