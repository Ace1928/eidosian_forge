import functools
from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfNumbersOrNone,\
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Ellipse, Wedge, String, STATE_DEFAULTS, ArcPath, Polygon, Rect, PolyLine, Line
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab.graphics.charts.textlabels import Label
from reportlab import cmp
from reportlab.graphics.charts.utils3d import _getShaded, _2rad, _360, _180_pi
class WedgeProperties(PropHolder):
    """This holds descriptive information about the wedges in a pie chart.

    It is not to be confused with the 'wedge itself'; this just holds
    a recipe for how to format one, and does not allow you to hack the
    angles.  It can format a genuine Wedge object for you with its
    format method.
    """
    _attrMap = AttrMap(strokeWidth=AttrMapValue(isNumber, desc='Width of the wedge border'), fillColor=AttrMapValue(isColorOrNone, desc='Filling color of the wedge'), strokeColor=AttrMapValue(isColorOrNone, desc='Color of the wedge border'), strokeDashArray=AttrMapValue(isListOfNumbersOrNone, desc='Style of the wedge border, expressed as a list of lengths of alternating dashes and blanks'), strokeLineCap=AttrMapValue(OneOf(0, 1, 2), desc='Line cap 0=butt, 1=round & 2=square'), strokeLineJoin=AttrMapValue(OneOf(0, 1, 2), desc='Line join 0=miter, 1=round & 2=bevel'), strokeMiterLimit=AttrMapValue(isNumber, desc='Miter limit control miter line joins'), popout=AttrMapValue(isNumber, desc='How far of centre a wedge to pop'), fontName=AttrMapValue(isString, desc='Name of the font of the label text'), fontSize=AttrMapValue(isNumber, desc='Size of the font of the label text in points'), fontColor=AttrMapValue(isColorOrNone, desc='Color of the font of the label text'), labelRadius=AttrMapValue(isNumber, desc='Distance between the center of the label box and the center of the pie, expressed in times the radius of the pie'), label_dx=AttrMapValue(isNumber, desc='X Offset of the label'), label_dy=AttrMapValue(isNumber, desc='Y Offset of the label'), label_angle=AttrMapValue(isNumber, desc='Angle of the label, default (0) is horizontal, 90 is vertical, 180 is upside down'), label_boxAnchor=AttrMapValue(isBoxAnchor, desc='Anchoring point of the label'), label_boxStrokeColor=AttrMapValue(isColorOrNone, desc='Border color for the label box'), label_boxStrokeWidth=AttrMapValue(isNumber, desc='Border width for the label box'), label_boxFillColor=AttrMapValue(isColorOrNone, desc='Filling color of the label box'), label_strokeColor=AttrMapValue(isColorOrNone, desc='Border color for the label text'), label_strokeWidth=AttrMapValue(isNumber, desc='Border width for the label text'), label_text=AttrMapValue(isStringOrNone, desc='Text of the label'), label_leading=AttrMapValue(isNumberOrNone, desc=''), label_width=AttrMapValue(isNumberOrNone, desc='Width of the label'), label_maxWidth=AttrMapValue(isNumberOrNone, desc='Maximum width the label can grow to'), label_height=AttrMapValue(isNumberOrNone, desc='Height of the label'), label_textAnchor=AttrMapValue(isTextAnchor, desc='Maximum height the label can grow to'), label_visible=AttrMapValue(isBoolean, desc='True if the label is to be drawn'), label_topPadding=AttrMapValue(isNumber, 'Padding at top of box'), label_leftPadding=AttrMapValue(isNumber, 'Padding at left of box'), label_rightPadding=AttrMapValue(isNumber, 'Padding at right of box'), label_bottomPadding=AttrMapValue(isNumber, 'Padding at bottom of box'), label_simple_pointer=AttrMapValue(isBoolean, 'Set to True for simple pointers'), label_pointer_strokeColor=AttrMapValue(isColorOrNone, desc='Color of indicator line'), label_pointer_strokeWidth=AttrMapValue(isNumber, desc='StrokeWidth of indicator line'), label_pointer_elbowLength=AttrMapValue(isNumber, desc='Length of final indicator line segment'), label_pointer_edgePad=AttrMapValue(isNumber, desc='pad between pointer label and box'), label_pointer_piePad=AttrMapValue(isNumber, desc='pad between pointer label and pie'), swatchMarker=AttrMapValue(NoneOr(isSymbol), desc="None or makeMarker('Diamond') ...", advancedUsage=1), visible=AttrMapValue(isBoolean, 'Set to false to skip displaying'), shadingAmount=AttrMapValue(isNumberOrNone, desc='amount by which to shade fillColor'), shadingAngle=AttrMapValue(isNumber, desc='shading changes at multiple of this angle (in degrees)'), shadingDirection=AttrMapValue(OneOf('normal', 'anti'), desc='Whether shading is at start or end of wedge/sector'), shadingKind=AttrMapValue(OneOf(None, 'lighten', 'darken'), desc='use colors.Whiter or Blacker'))

    def __init__(self):
        self.strokeWidth = 0
        self.fillColor = None
        self.strokeColor = STATE_DEFAULTS['strokeColor']
        self.strokeDashArray = STATE_DEFAULTS['strokeDashArray']
        self.strokeLineJoin = 1
        self.strokeLineCap = 0
        self.strokeMiterLimit = 0
        self.popout = 0
        self.fontName = STATE_DEFAULTS['fontName']
        self.fontSize = STATE_DEFAULTS['fontSize']
        self.fontColor = STATE_DEFAULTS['fillColor']
        self.labelRadius = 1.2
        self.label_dx = self.label_dy = self.label_angle = 0
        self.label_text = None
        self.label_topPadding = self.label_leftPadding = self.label_rightPadding = self.label_bottomPadding = 0
        self.label_boxAnchor = 'autox'
        self.label_boxStrokeColor = None
        self.label_boxStrokeWidth = 0.5
        self.label_boxFillColor = None
        self.label_strokeColor = None
        self.label_strokeWidth = 0.1
        self.label_leading = self.label_width = self.label_maxWidth = self.label_height = None
        self.label_textAnchor = 'start'
        self.label_simple_pointer = 0
        self.label_visible = 1
        self.label_pointer_strokeColor = colors.black
        self.label_pointer_strokeWidth = 0.5
        self.label_pointer_elbowLength = 3
        self.label_pointer_edgePad = 2
        self.label_pointer_piePad = 3
        self.visible = 1
        self.shadingKind = None
        self.shadingAmount = 0.5
        self.shadingAngle = 2.0137
        self.shadingDirection = 'normal'