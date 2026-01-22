from fontTools.pens.basePen import AbstractPen, DecomposingPen
from fontTools.pens.pointPen import AbstractPointPen, DecomposingPointPen
class RecordingPointPen(AbstractPointPen):
    """PointPen recording operations that can be accessed or replayed.

    The recording can be accessed as pen.value; or replayed using
    pointPen.replay(otherPointPen).

    :Example:

            from defcon import Font
            from fontTools.pens.recordingPen import RecordingPointPen

            glyph_name = 'a'
            font_path = 'MyFont.ufo'

            font = Font(font_path)
            glyph = font[glyph_name]

            pen = RecordingPointPen()
            glyph.drawPoints(pen)
            print(pen.value)

            new_glyph = font.newGlyph('b')
            pen.replay(new_glyph.getPointPen())
    """

    def __init__(self):
        self.value = []

    def beginPath(self, identifier=None, **kwargs):
        if identifier is not None:
            kwargs['identifier'] = identifier
        self.value.append(('beginPath', (), kwargs))

    def endPath(self):
        self.value.append(('endPath', (), {}))

    def addPoint(self, pt, segmentType=None, smooth=False, name=None, identifier=None, **kwargs):
        if identifier is not None:
            kwargs['identifier'] = identifier
        self.value.append(('addPoint', (pt, segmentType, smooth, name), kwargs))

    def addComponent(self, baseGlyphName, transformation, identifier=None, **kwargs):
        if identifier is not None:
            kwargs['identifier'] = identifier
        self.value.append(('addComponent', (baseGlyphName, transformation), kwargs))

    def addVarComponent(self, baseGlyphName, transformation, location, identifier=None, **kwargs):
        if identifier is not None:
            kwargs['identifier'] = identifier
        self.value.append(('addVarComponent', (baseGlyphName, transformation, location), kwargs))

    def replay(self, pointPen):
        for operator, args, kwargs in self.value:
            getattr(pointPen, operator)(*args, **kwargs)
    drawPoints = replay