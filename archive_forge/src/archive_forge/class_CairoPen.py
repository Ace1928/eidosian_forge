from fontTools.pens.basePen import BasePen
class CairoPen(BasePen):
    """Pen to draw to a Cairo graphics library context."""

    def __init__(self, glyphSet, context):
        BasePen.__init__(self, glyphSet)
        self.context = context

    def _moveTo(self, p):
        self.context.move_to(*p)

    def _lineTo(self, p):
        self.context.line_to(*p)

    def _curveToOne(self, p1, p2, p3):
        self.context.curve_to(*p1, *p2, *p3)

    def _closePath(self):
        self.context.close_path()