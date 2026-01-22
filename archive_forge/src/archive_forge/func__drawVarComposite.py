from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from copy import copy
from types import SimpleNamespace
from fontTools.misc.fixedTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.misc.transform import Transform
from fontTools.pens.transformPen import TransformPen, TransformPointPen
from fontTools.pens.recordingPen import (
def _drawVarComposite(self, glyph, pen, isPointPen):
    from fontTools.ttLib.tables._g_l_y_f import VarComponentFlags, VAR_COMPONENT_TRANSFORM_MAPPING
    for comp in glyph.components:
        with self.glyphSet.pushLocation(comp.location, comp.flags & VarComponentFlags.RESET_UNSPECIFIED_AXES):
            try:
                pen.addVarComponent(comp.glyphName, comp.transform, self.glyphSet.rawLocation)
            except AttributeError:
                t = comp.transform.toTransform()
                if isPointPen:
                    tPen = TransformPointPen(pen, t)
                    self.glyphSet[comp.glyphName].drawPoints(tPen)
                else:
                    tPen = TransformPen(pen, t)
                    self.glyphSet[comp.glyphName].draw(tPen)