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
def _buildPaintCallbacks():
    return {(BuildCallback.BEFORE_BUILD, ot.Paint, ot.PaintFormat.PaintRadialGradient): _beforeBuildPaintRadialGradient, (BuildCallback.BEFORE_BUILD, ot.Paint, ot.PaintFormat.PaintVarRadialGradient): _beforeBuildPaintRadialGradient, (BuildCallback.CREATE_DEFAULT, ot.ColorStop): _defaultColorStop, (BuildCallback.CREATE_DEFAULT, ot.VarColorStop): _defaultVarColorStop, (BuildCallback.CREATE_DEFAULT, ot.ColorLine): _defaultColorLine, (BuildCallback.CREATE_DEFAULT, ot.VarColorLine): _defaultVarColorLine, (BuildCallback.CREATE_DEFAULT, ot.Paint, ot.PaintFormat.PaintSolid): _defaultPaintSolid, (BuildCallback.CREATE_DEFAULT, ot.Paint, ot.PaintFormat.PaintVarSolid): _defaultPaintSolid}