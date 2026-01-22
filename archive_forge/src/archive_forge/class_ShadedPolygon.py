from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isColorOrNone, isBoolean, isListOfNumbers, OneOf, isListOfColors, isNumberOrNone
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, LineShape, definePath, EmptyClipPath
from reportlab.graphics.widgetbase import Widget
class ShadedPolygon(Widget, LineShape):
    _attrMap = AttrMap(BASE=LineShape, angle=AttrMapValue(isNumber, desc='Shading angle'), fillColorStart=AttrMapValue(isColorOrNone), fillColorEnd=AttrMapValue(isColorOrNone), numShades=AttrMapValue(isNumber, desc='The number of interpolating colors.'), cylinderMode=AttrMapValue(isBoolean, desc='True if shading reverses in middle.'), points=AttrMapValue(isListOfNumbers))

    def __init__(self, **kw):
        self.angle = 90
        self.fillColorStart = colors.red
        self.fillColorEnd = colors.green
        self.cylinderMode = 0
        self.numShades = 50
        self.points = [-1, -1, 2, 2, 3, -1]
        LineShape.__init__(self, kw)

    def draw(self):
        P = self.points
        P = list(map(lambda i, P=P: (P[i], P[i + 1]), range(0, len(P), 2)))
        path = definePath([('moveTo',) + P[0]] + [('lineTo',) + x for x in P[1:]] + ['closePath'], fillColor=None, strokeColor=None)
        path.isClipPath = 1
        g = Group()
        g.add(path)
        angle = self.angle
        orientation = 'vertical'
        if angle == 180:
            angle = 0
        elif angle in (90, 270):
            orientation = 'horizontal'
            angle = 0
        rect = ShadedRect(strokeWidth=0, strokeColor=None, orientation=orientation)
        for k in ('fillColorStart', 'fillColorEnd', 'numShades', 'cylinderMode'):
            setattr(rect, k, getattr(self, k))
        g.add(rotatedEnclosingRect(P, angle, rect))
        g.add(EmptyClipPath)
        path = path.copy()
        path.isClipPath = 0
        path.strokeColor = self.strokeColor
        path.strokeWidth = self.strokeWidth
        g.add(path)
        return g