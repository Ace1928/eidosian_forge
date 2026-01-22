from math import sqrt, degrees, atan
from fontTools.pens.basePen import BasePen, OpenContourError
from fontTools.pens.momentsPen import MomentsPen
class StatisticsPen(StatisticsBase, MomentsPen):
    """Pen calculating area, center of mass, variance and
    standard-deviation, covariance and correlation, and slant,
    of glyph shapes.

    Note that if the glyph shape is self-intersecting, the values
    are not correct (but well-defined). Moreover, area will be
    negative if contour directions are clockwise."""

    def __init__(self, glyphset=None):
        MomentsPen.__init__(self, glyphset=glyphset)
        StatisticsBase.__init__(self)

    def _closePath(self):
        MomentsPen._closePath(self)
        self._update()

    def _update(self):
        area = self.area
        if not area:
            self._zero()
            return
        self.meanX = meanX = self.momentX / area
        self.meanY = meanY = self.momentY / area
        self.varianceX = self.momentXX / area - meanX * meanX
        self.varianceY = self.momentYY / area - meanY * meanY
        self.covariance = self.momentXY / area - meanX * meanY
        StatisticsBase._update(self)