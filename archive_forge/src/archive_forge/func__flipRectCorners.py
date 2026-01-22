from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isColorOrNone, isBoolean, isListOfNumbers, OneOf, isListOfColors, isNumberOrNone
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, LineShape, definePath, EmptyClipPath
from reportlab.graphics.widgetbase import Widget
def _flipRectCorners(self):
    """Flip rectangle's corners if width or height is negative."""
    x, y, width, height, fillColorStart, fillColorEnd = (self.x, self.y, self.width, self.height, self.fillColorStart, self.fillColorEnd)
    if width < 0 and height > 0:
        x = x + width
        width = -width
        if self.orientation == 'vertical':
            fillColorStart, fillColorEnd = (fillColorEnd, fillColorStart)
    elif height < 0 and width > 0:
        y = y + height
        height = -height
        if self.orientation == 'horizontal':
            fillColorStart, fillColorEnd = (fillColorEnd, fillColorStart)
    elif height < 0 and height < 0:
        x = x + width
        width = -width
        y = y + height
        height = -height
    return (x, y, width, height, fillColorStart, fillColorEnd)