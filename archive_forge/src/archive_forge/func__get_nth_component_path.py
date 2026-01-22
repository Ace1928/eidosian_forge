import itertools
import logging
from typing import Callable, Iterable, Optional, Mapping
from fontTools.misc.roundTools import otRound
from fontTools.ttLib import ttFont
from fontTools.ttLib.tables import _g_l_y_f
from fontTools.ttLib.tables import _h_m_t_x
from fontTools.pens.ttGlyphPen import TTGlyphPen
import pathops
def _get_nth_component_path(index: int) -> pathops.Path:
    if index not in component_paths:
        component_paths[index] = skPathFromGlyphComponent(glyph.components[index], glyphSet)
    return component_paths[index]