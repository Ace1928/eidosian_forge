from array import array
from typing import Any, Callable, Dict, Optional, Tuple
from fontTools.misc.fixedTools import MAX_F2DOT14, floatToFixedToFloat
from fontTools.misc.loggingTools import LogMixin
from fontTools.pens.pointPen import AbstractPointPen
from fontTools.misc.roundTools import otRound
from fontTools.pens.basePen import LoggingPen, PenError
from fontTools.pens.transformPen import TransformPen, TransformPointPen
from fontTools.ttLib.tables import ttProgram
from fontTools.ttLib.tables._g_l_y_f import flagOnCurve, flagCubic
from fontTools.ttLib.tables._g_l_y_f import Glyph
from fontTools.ttLib.tables._g_l_y_f import GlyphComponent
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
from fontTools.ttLib.tables._g_l_y_f import dropImpliedOnCurvePoints
import math
class _TTGlyphBasePen:

    def __init__(self, glyphSet: Optional[Dict[str, Any]], handleOverflowingTransforms: bool=True) -> None:
        """
        Construct a new pen.

        Args:
            glyphSet (Dict[str, Any]): A glyphset object, used to resolve components.
            handleOverflowingTransforms (bool): See below.

        If ``handleOverflowingTransforms`` is True, the components' transform values
        are checked that they don't overflow the limits of a F2Dot14 number:
        -2.0 <= v < +2.0. If any transform value exceeds these, the composite
        glyph is decomposed.

        An exception to this rule is done for values that are very close to +2.0
        (both for consistency with the -2.0 case, and for the relative frequency
        these occur in real fonts). When almost +2.0 values occur (and all other
        values are within the range -2.0 <= x <= +2.0), they are clamped to the
        maximum positive value that can still be encoded as an F2Dot14: i.e.
        1.99993896484375.

        If False, no check is done and all components are translated unmodified
        into the glyf table, followed by an inevitable ``struct.error`` once an
        attempt is made to compile them.

        If both contours and components are present in a glyph, the components
        are decomposed.
        """
        self.glyphSet = glyphSet
        self.handleOverflowingTransforms = handleOverflowingTransforms
        self.init()

    def _decompose(self, glyphName: str, transformation: Tuple[float, float, float, float, float, float]):
        tpen = self.transformPen(self, transformation)
        getattr(self.glyphSet[glyphName], self.drawMethod)(tpen)

    def _isClosed(self):
        """
        Check if the current path is closed.
        """
        raise NotImplementedError

    def init(self) -> None:
        self.points = []
        self.endPts = []
        self.types = []
        self.components = []

    def addComponent(self, baseGlyphName: str, transformation: Tuple[float, float, float, float, float, float], identifier: Optional[str]=None, **kwargs: Any) -> None:
        """
        Add a sub glyph.
        """
        self.components.append((baseGlyphName, transformation))

    def _buildComponents(self, componentFlags):
        if self.handleOverflowingTransforms:
            overflowing = any((s > 2 or s < -2 for glyphName, transformation in self.components for s in transformation[:4]))
        components = []
        for glyphName, transformation in self.components:
            if glyphName not in self.glyphSet:
                self.log.warning(f"skipped non-existing component '{glyphName}'")
                continue
            if self.points or (self.handleOverflowingTransforms and overflowing):
                self._decompose(glyphName, transformation)
                continue
            component = GlyphComponent()
            component.glyphName = glyphName
            component.x, component.y = (otRound(v) for v in transformation[4:])
            transformation = tuple((floatToFixedToFloat(v, 14) for v in transformation[:4]))
            if transformation != (1, 0, 0, 1):
                if self.handleOverflowingTransforms and any((MAX_F2DOT14 < s <= 2 for s in transformation)):
                    transformation = tuple((MAX_F2DOT14 if MAX_F2DOT14 < s <= 2 else s for s in transformation))
                component.transform = (transformation[:2], transformation[2:])
            component.flags = componentFlags
            components.append(component)
        return components

    def glyph(self, componentFlags: int=4, dropImpliedOnCurves: bool=False, *, round: Callable[[float], int]=otRound) -> Glyph:
        """
        Returns a :py:class:`~._g_l_y_f.Glyph` object representing the glyph.

        Args:
            componentFlags: Flags to use for component glyphs. (default: 0x04)

            dropImpliedOnCurves: Whether to remove implied-oncurve points. (default: False)
        """
        if not self._isClosed():
            raise PenError("Didn't close last contour.")
        components = self._buildComponents(componentFlags)
        glyph = Glyph()
        glyph.coordinates = GlyphCoordinates(self.points)
        glyph.endPtsOfContours = self.endPts
        glyph.flags = array('B', self.types)
        self.init()
        if components:
            glyph.components = components
            glyph.numberOfContours = -1
        else:
            glyph.numberOfContours = len(glyph.endPtsOfContours)
            glyph.program = ttProgram.Program()
            glyph.program.fromBytecode(b'')
            if dropImpliedOnCurves:
                dropImpliedOnCurvePoints(glyph)
            glyph.coordinates.toInt(round=round)
        return glyph