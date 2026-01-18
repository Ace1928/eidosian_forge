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