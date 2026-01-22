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
class Pie3d(Pie):
    _attrMap = AttrMap(BASE=Pie, perspective=AttrMapValue(isNumber, desc='A flattening parameter.'), depth_3d=AttrMapValue(isNumber, desc='depth of the pie.'), angle_3d=AttrMapValue(isNumber, desc='The view angle.'))
    perspective = 70
    depth_3d = 25
    angle_3d = 180

    def _popout(self, i):
        return self._sl3d[i].not360 and self.slices[i].popout or 0

    def CX(self, i, d):
        return self._cx + (d and self._xdepth_3d or 0) + self._popout(i) * cos(_2rad(self._sl3d[i].mid))

    def CY(self, i, d):
        return self._cy + (d and self._ydepth_3d or 0) + self._popout(i) * sin(_2rad(self._sl3d[i].mid))

    def OX(self, i, o, d):
        return self.CX(i, d) + self._radiusx * cos(_2rad(o))

    def OY(self, i, o, d):
        return self.CY(i, d) + self._radiusy * sin(_2rad(o))

    def rad_dist(self, a):
        _3dva = self._3dva
        return min(abs(a - _3dva), abs(a - _3dva + 360))

    def __init__(self):
        Pie.__init__(self)
        self.slices = TypedPropertyCollection(Wedge3dProperties)
        self.slices[0].fillColor = colors.darkcyan
        self.slices[1].fillColor = colors.blueviolet
        self.slices[2].fillColor = colors.blue
        self.slices[3].fillColor = colors.cyan
        self.slices[4].fillColor = colors.azure
        self.slices[5].fillColor = colors.crimson
        self.slices[6].fillColor = colors.darkviolet
        self.xradius = self.yradius = None
        self.width = 300
        self.height = 200
        self.data = [12.5, 20.1, 2.0, 22.0, 5.0, 18.0, 13.0]

    def _fillSide(self, L, i, angle, strokeColor, strokeWidth, fillColor):
        rd = self.rad_dist(angle)
        if rd < self.rad_dist(self._sl3d[i].mid):
            p = [self.CX(i, 0), self.CY(i, 0), self.CX(i, 1), self.CY(i, 1), self.OX(i, angle, 1), self.OY(i, angle, 1), self.OX(i, angle, 0), self.OY(i, angle, 0)]
            L.append((rd, Polygon(p, strokeColor=strokeColor, fillColor=fillColor, strokeWidth=strokeWidth, strokeLineJoin=1)))

    def draw(self):
        slices = self.slices
        _3d_angle = self.angle_3d
        _3dva = self._3dva = _360(_3d_angle + 90)
        a0 = _2rad(_3dva)
        depth_3d = self.depth_3d
        self._xdepth_3d = cos(a0) * depth_3d
        self._ydepth_3d = sin(a0) * depth_3d
        self._cx = self.x + self.width / 2.0
        self._cy = self.y + (self.height - self._ydepth_3d) / 2.0
        radiusx = radiusy = self._cx - self.x
        if self.xradius:
            radiusx = self.xradius
        if self.yradius:
            radiusy = self.yradius
        self._radiusx = radiusx
        self._radiusy = radiusy = (1.0 - self.perspective / 100.0) * radiusy
        data = self.normalizeData()
        sum = self._sum
        CX = self.CX
        CY = self.CY
        OX = self.OX
        OY = self.OY
        rad_dist = self.rad_dist
        _fillSide = self._fillSide
        self._seriesCount = n = len(data)
        _sl3d = self._sl3d = []
        g = Group()
        last = _360(self.startAngle)
        a0 = self.direction == 'clockwise' and -1 or 1
        for v in data:
            v *= a0
            angle1, angle0 = (last, v + last)
            last = angle0
            if a0 > 0:
                angle0, angle1 = (angle1, angle0)
            _sl3d.append(_SL3D(angle0, angle1))
        labels = _fixLabels(self.labels, n)
        a0 = _3d_angle
        a1 = _3d_angle + 180
        T = []
        S = []
        L = []

        class WedgeLabel3d(WedgeLabel):
            _ydepth_3d = self._ydepth_3d

            def _checkDXY(self, ba):
                if ba[0] == 'n':
                    if not hasattr(self, '_ody'):
                        self._ody = self.dy
                        self.dy = -self._ody + self._ydepth_3d
        checkLabelOverlap = self.checkLabelOverlap
        for i in range(n):
            style = slices[i]
            if not style.visible:
                continue
            sl = _sl3d[i]
            lo = angle0 = sl.lo
            hi = angle1 = sl.hi
            aa = abs(hi - lo)
            if aa < _ANGLELO:
                continue
            fillColor = _getShaded(style.fillColor, style.fillColorShaded, style.shading)
            strokeColor = _getShaded(style.strokeColor, style.strokeColorShaded, style.shading) or fillColor
            strokeWidth = style.strokeWidth
            cx0 = CX(i, 0)
            cy0 = CY(i, 0)
            cx1 = CX(i, 1)
            cy1 = CY(i, 1)
            if depth_3d:
                g.add(Wedge(cx1, cy1, radiusx, lo, hi, yradius=radiusy, strokeColor=strokeColor, strokeWidth=strokeWidth, fillColor=fillColor, strokeLineJoin=1))
                if lo < a0 < hi:
                    angle0 = a0
                if lo < a1 < hi:
                    angle1 = a1
                p = ArcPath(strokeColor=strokeColor, fillColor=fillColor, strokeWidth=strokeWidth, strokeLineJoin=1)
                p.addArc(cx1, cy1, radiusx, angle0, angle1, yradius=radiusy, moveTo=1)
                p.lineTo(OX(i, angle1, 0), OY(i, angle1, 0))
                p.addArc(cx0, cy0, radiusx, angle0, angle1, yradius=radiusy, reverse=1)
                p.closePath()
                if angle0 <= _3dva and angle1 >= _3dva:
                    rd = 0
                else:
                    rd = min(rad_dist(angle0), rad_dist(angle1))
                S.append((rd, p))
                _fillSide(S, i, lo, strokeColor, strokeWidth, fillColor)
                _fillSide(S, i, hi, strokeColor, strokeWidth, fillColor)
            fillColor = style.fillColor
            strokeColor = style.strokeColor or fillColor
            T.append(Wedge(cx0, cy0, radiusx, lo, hi, yradius=radiusy, strokeColor=strokeColor, strokeWidth=strokeWidth, fillColor=fillColor, strokeLineJoin=1))
            if aa >= _ANGLEHI:
                theWedge = Ellipse(cx0, cy0, radiusx, radiusy, strokeColor=strokeColor, strokeWidth=strokeWidth, fillColor=fillColor, strokeLineJoin=1)
            else:
                theWedge = Wedge(cx0, cy0, radiusx, lo, hi, yradius=radiusy, strokeColor=strokeColor, strokeWidth=strokeWidth, fillColor=fillColor, strokeLineJoin=1)
            T.append(theWedge)
            text = labels[i]
            if style.label_visible and text:
                rat = style.labelRadius
                self._radiusx *= rat
                self._radiusy *= rat
                mid = sl.mid
                labelX = OX(i, mid, 0)
                labelY = OY(i, mid, 0)
                l = _addWedgeLabel(self, text, mid, labelX, labelY, style, labelClass=WedgeLabel3d)
                L.append(l)
                if checkLabelOverlap:
                    l._origdata = {'x': labelX, 'y': labelY, 'angle': mid, 'rx': self._radiusx, 'ry': self._radiusy, 'cx': CX(i, 0), 'cy': CY(i, 0), 'bounds': l.getBounds()}
                self._radiusx = radiusx
                self._radiusy = radiusy
        S.sort(key=_keyS3D)
        if checkLabelOverlap and L:
            fixLabelOverlaps(L, self.sideLabels)
        for x in [s[1] for s in S] + T + L:
            g.add(x)
        return g

    def demo(self):
        d = Drawing(200, 100)
        pc = Pie()
        pc.x = 50
        pc.y = 10
        pc.width = 100
        pc.height = 80
        pc.data = [10, 20, 30, 40, 50, 60]
        pc.labels = ['a', 'b', 'c', 'd', 'e', 'f']
        pc.slices.strokeWidth = 0.5
        pc.slices[3].popout = 10
        pc.slices[3].strokeWidth = 2
        pc.slices[3].strokeDashArray = [2, 2]
        pc.slices[3].labelRadius = 1.75
        pc.slices[3].fontColor = colors.red
        pc.slices[0].fillColor = colors.darkcyan
        pc.slices[1].fillColor = colors.blueviolet
        pc.slices[2].fillColor = colors.blue
        pc.slices[3].fillColor = colors.cyan
        pc.slices[4].fillColor = colors.aquamarine
        pc.slices[5].fillColor = colors.cadetblue
        pc.slices[6].fillColor = colors.lightcoral
        self.slices[1].visible = 0
        self.slices[3].visible = 1
        self.slices[4].visible = 1
        self.slices[5].visible = 1
        self.slices[6].visible = 0
        d.add(pc)
        return d