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
class Pie(AbstractPieChart):
    _attrMap = AttrMap(BASE=AbstractPieChart, data=AttrMapValue(isListOfNumbers, desc='List of numbers defining wedge sizes; need not sum to 1'), labels=AttrMapValue(isListOfStringsOrNone, desc='Optional list of labels to use for each data point'), startAngle=AttrMapValue(isNumber, desc='Angle of first slice; 0 is due East'), direction=AttrMapValue(OneOf('clockwise', 'anticlockwise'), desc="'clockwise' or 'anticlockwise'"), slices=AttrMapValue(None, desc='Collection of wedge descriptor objects'), simpleLabels=AttrMapValue(isBoolean, desc='If true(default) use a simple String not an advanced WedgeLabel. A WedgeLabel is customisable using the properties prefixed label_ in the collection slices.'), other_threshold=AttrMapValue(isNumber, desc='A value for doing threshholding, not used yet.', advancedUsage=1), checkLabelOverlap=AttrMapValue(EitherOr((isNumberInRange(0.05, 1), isBoolean)), desc='If true check and attempt to fix\n standard label overlaps(default off)', advancedUsage=1), pointerLabelMode=AttrMapValue(OneOf(None, 'LeftRight', 'LeftAndRight'), desc='', advancedUsage=1), sameRadii=AttrMapValue(isBoolean, desc='If true make x/y radii the same(default off)', advancedUsage=1), orderMode=AttrMapValue(OneOf('fixed', 'alternate'), advancedUsage=1), xradius=AttrMapValue(isNumberOrNone, desc='X direction Radius'), yradius=AttrMapValue(isNumberOrNone, desc='Y direction Radius'), innerRadiusFraction=AttrMapValue(isNumberOrNone, desc='fraction of radii to start wedges at'), wedgeRecord=AttrMapValue(None, desc='callable(wedge,*args,**kwds)', advancedUsage=1), sideLabels=AttrMapValue(isBoolean, desc='If true attempt to make piechart with labels along side and pointers'), sideLabelsOffset=AttrMapValue(isNumber, desc='The fraction of the pie width that the labels are situated at from the edges of the pie'), labelClass=AttrMapValue(NoneOr(isCallable), desc='A class factory to use for non simple labels'))
    other_threshold = None

    def __init__(self, **kwd):
        PlotArea.__init__(self)
        self.x = 0
        self.y = 0
        self.width = 100
        self.height = 100
        self.data = [1, 2.3, 1.7, 4.2]
        self.labels = None
        self.startAngle = 90
        self.direction = 'clockwise'
        self.simpleLabels = 1
        self.checkLabelOverlap = 0
        self.pointerLabelMode = None
        self.sameRadii = False
        self.orderMode = 'fixed'
        self.xradius = self.yradius = self.innerRadiusFraction = None
        self.sideLabels = 0
        self.sideLabelsOffset = 0.1
        self.slices = TypedPropertyCollection(WedgeProperties)
        self.slices[0].fillColor = colors.darkcyan
        self.slices[1].fillColor = colors.blueviolet
        self.slices[2].fillColor = colors.blue
        self.slices[3].fillColor = colors.cyan
        self.slices[4].fillColor = colors.pink
        self.slices[5].fillColor = colors.magenta
        self.slices[6].fillColor = colors.yellow

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
        d.add(pc)
        return d

    def makePointerLabels(self, angles, plMode):

        class PL:

            def __init__(self, centerx, centery, xradius, yradius, data, lu=0, ru=0):
                self.centerx = centerx
                self.centery = centery
                self.xradius = xradius
                self.yradius = yradius
                self.data = data
                self.lu = lu
                self.ru = ru
        labelX = self.width - 2
        labelY = self.height
        n = nr = nl = maxW = sumH = 0
        styleCount = len(self.slices)
        L = []
        L_add = L.append
        refArcs = _makeSideArcDefs(self.startAngle, self.direction)
        for i, A in angles:
            if A[1] is None:
                continue
            sn = self.getSeriesName(i, '')
            if not sn:
                continue
            style = self.slices[i % styleCount]
            if not style.label_visible or not style.visible:
                continue
            n += 1
            l = _addWedgeLabel(self, sn, 180, labelX, labelY, style)
            L_add(l)
            b = l.getBounds()
            w = b[2] - b[0]
            h = b[3] - b[1]
            ri = [(a[0], intervalIntersection(A, (a[1], a[2]))) for a in refArcs]
            li = _findLargestArc(ri, 0)
            ri = _findLargestArc(ri, 1)
            if li and ri:
                if plMode == 'LeftAndRight':
                    if li[1] - li[0] < ri[1] - ri[0]:
                        li = None
                    else:
                        ri = None
                elif li[1] - li[0] < 0.02 * (ri[1] - ri[0]):
                    li = None
                elif (li[1] - li[0]) * 0.02 > ri[1] - ri[0]:
                    ri = None
            if ri:
                nr += 1
            if li:
                nl += 1
            l._origdata = dict(bounds=b, width=w, height=h, li=li, ri=ri, index=i, edgePad=style.label_pointer_edgePad, piePad=style.label_pointer_piePad, elbowLength=style.label_pointer_elbowLength)
            maxW = max(w, maxW)
            sumH += h + 2
        if not n:
            xradius = self.width * 0.5
            yradius = self.height * 0.5
            centerx = self.x + xradius
            centery = self.y + yradius
            if self.xradius:
                xradius = self.xradius
            if self.yradius:
                yradius = self.yradius
            if self.sameRadii:
                xradius = yradius = min(xradius, yradius)
            return PL(centerx, centery, xradius, yradius, [])
        aonR = nr == n
        if sumH < self.height and (aonR or nl == n):
            side = int(aonR)
        else:
            side = None
        G, lu, ru, mel = _fixPointerLabels(len(angles), L, self.x, self.y, self.width, self.height, side=side)
        if plMode == 'LeftAndRight':
            lu = ru = max(lu, ru)
        x0 = self.x + lu
        x1 = self.x + self.width - ru
        xradius = (x1 - x0) * 0.5
        yradius = self.height * 0.5 - mel
        centerx = x0 + xradius
        centery = self.y + yradius + mel
        if self.xradius:
            xradius = self.xradius
        if self.yradius:
            yradius = self.yradius
        if self.sameRadii:
            xradius = yradius = min(xradius, yradius)
        return PL(centerx, centery, xradius, yradius, G, lu, ru)

    def normalizeData(self, keepData=False):
        data = list(map(abs, self.data))
        s = self._sum = float(sum(data))
        f = 360.0 / s if s != 0 else 1
        if keepData:
            return [AngleData(f * x, x) for x in data]
        else:
            return [f * x for x in data]

    def makeAngles(self):
        wr = getattr(self, 'wedgeRecord', None)
        if self.sideLabels:
            startAngle = theta0(self.data, self.direction)
            self.slices.label_visible = 1
        else:
            startAngle = self.startAngle % 360
        whichWay = self.direction == 'clockwise' and -1 or 1
        D = [a for a in enumerate(self.normalizeData(keepData=wr))]
        if self.orderMode == 'alternate' and (not self.sideLabels):
            W = [a for a in D if abs(a[1]) >= 1e-05]
            W.sort(key=_arcCF)
            T = [[], []]
            i = 0
            while W:
                if i < 2:
                    a = W.pop(0)
                else:
                    a = W.pop(-1)
                T[i % 2].append(a)
                i += 1
                i %= 4
            T[1].reverse()
            D = T[0] + T[1] + [a for a in D if abs(a[1]) < 1e-05]
        A = []
        a = A.append
        for i, angle in D:
            endAngle = startAngle + angle * whichWay
            if abs(angle) >= _ANGLELO:
                if startAngle >= endAngle:
                    aa = (endAngle, startAngle)
                else:
                    aa = (startAngle, endAngle)
            else:
                aa = (startAngle, None)
            if wr:
                aa = (AngleData(aa[0], angle._data), aa[1])
            startAngle = endAngle
            a((i, aa))
        return A

    def makeWedges(self):
        angles = self.makeAngles()
        halfAngles = []
        for i, (a1, a2) in angles:
            if a2 is None:
                halfAngle = a1
            else:
                halfAngle = 0.5 * (a2 + a1)
            halfAngles.append(halfAngle)
        sideLabels = self.sideLabels
        n = len(angles)
        labels = _fixLabels(self.labels, n)
        wr = getattr(self, 'wedgeRecord', None)
        self._seriesCount = n
        styleCount = len(self.slices)
        plMode = self.pointerLabelMode
        if sideLabels:
            plMode = None
        if plMode:
            checkLabelOverlap = False
            PL = self.makePointerLabels(angles, plMode)
            xradius = PL.xradius
            yradius = PL.yradius
            centerx = PL.centerx
            centery = PL.centery
            PL_data = PL.data
            gSN = lambda i: ''
        else:
            xradius = self.width * 0.5
            yradius = self.height * 0.5
            centerx = self.x + xradius
            centery = self.y + yradius
            if self.xradius:
                xradius = self.xradius
            if self.yradius:
                yradius = self.yradius
            if self.sameRadii:
                xradius = yradius = min(xradius, yradius)
            checkLabelOverlap = self.checkLabelOverlap
            gSN = lambda i: self.getSeriesName(i, '')
        g = Group()
        g_add = g.add
        L = []
        L_add = L.append
        innerRadiusFraction = self.innerRadiusFraction
        for i, (a1, a2) in angles:
            if a2 is None:
                continue
            wedgeStyle = self.slices[i % styleCount]
            if not wedgeStyle.visible:
                continue
            aa = abs(a2 - a1)
            cx, cy = (centerx, centery)
            text = gSN(i)
            popout = wedgeStyle.popout
            if text or popout:
                averageAngle = (a1 + a2) / 2.0
                aveAngleRadians = averageAngle / _180_pi
                cosAA = cos(aveAngleRadians)
                sinAA = sin(aveAngleRadians)
                if popout and aa < _ANGLEHI:
                    cx = centerx + popout * cosAA
                    cy = centery + popout * sinAA
            if innerRadiusFraction:
                theWedge = Wedge(cx, cy, xradius, a1, a2, yradius=yradius, radius1=xradius * innerRadiusFraction, yradius1=yradius * innerRadiusFraction)
            elif aa >= _ANGLEHI:
                theWedge = Ellipse(cx, cy, xradius, yradius)
            else:
                theWedge = Wedge(cx, cy, xradius, a1, a2, yradius=yradius)
            theWedge.fillColor = wedgeStyle.fillColor
            theWedge.strokeColor = wedgeStyle.strokeColor
            theWedge.strokeWidth = wedgeStyle.strokeWidth
            theWedge.strokeLineJoin = wedgeStyle.strokeLineJoin
            theWedge.strokeLineCap = wedgeStyle.strokeLineCap
            theWedge.strokeMiterLimit = wedgeStyle.strokeMiterLimit
            theWedge.strokeDashArray = wedgeStyle.strokeDashArray
            shader = wedgeStyle.shadingKind
            if shader:
                nshades = aa / float(wedgeStyle.shadingAngle)
                if nshades > 1:
                    shader = colors.Whiter if shader == 'lighten' else colors.Blacker
                    nshades = 1 + int(nshades)
                    shadingAmount = 1 - wedgeStyle.shadingAmount
                    if wedgeStyle.shadingDirection == 'normal':
                        dsh = (1 - shadingAmount) / float(nshades - 1)
                        shf1 = shadingAmount
                    else:
                        dsh = (shadingAmount - 1) / float(nshades - 1)
                        shf1 = 1
                    shda = (a2 - a1) / float(nshades)
                    shsc = wedgeStyle.fillColor
                    theWedge.fillColor = None
                    for ish in range(nshades):
                        sha1 = a1 + ish * shda
                        sha2 = a1 + (ish + 1) * shda
                        shc = shader(shsc, shf1 + dsh * ish)
                        if innerRadiusFraction:
                            shWedge = Wedge(cx, cy, xradius, sha1, sha2, yradius=yradius, radius1=xradius * innerRadiusFraction, yradius1=yradius * innerRadiusFraction)
                        else:
                            shWedge = Wedge(cx, cy, xradius, sha1, sha2, yradius=yradius)
                        shWedge.fillColor = shc
                        shWedge.strokeColor = None
                        shWedge.strokeWidth = 0
                        g_add(shWedge)
            g_add(theWedge)
            if wr:
                wr(theWedge, value=a1._data, label=text)
            if wedgeStyle.label_visible:
                if not sideLabels:
                    if text:
                        labelRadius = wedgeStyle.labelRadius
                        rx = xradius * labelRadius
                        ry = yradius * labelRadius
                        labelX = cx + rx * cosAA
                        labelY = cy + ry * sinAA
                        l = _addWedgeLabel(self, text, averageAngle, labelX, labelY, wedgeStyle)
                        L_add(l)
                        if not plMode and l._simple_pointer:
                            l._aax = cx + xradius * cosAA
                            l._aay = cy + yradius * sinAA
                        if checkLabelOverlap:
                            l._origdata = {'x': labelX, 'y': labelY, 'angle': averageAngle, 'rx': rx, 'ry': ry, 'cx': cx, 'cy': cy, 'bounds': l.getBounds(), 'angles': (a1, a2)}
                    elif plMode and PL_data:
                        l = PL_data[i]
                        if l:
                            data = l._origdata
                            sinM = data['smid']
                            cosM = data['cmid']
                            lX = cx + xradius * cosM
                            lY = cy + yradius * sinM
                            lpel = wedgeStyle.label_pointer_elbowLength
                            lXi = lX + lpel * cosM
                            lYi = lY + lpel * sinM
                            L_add(PolyLine((lX, lY, lXi, lYi, l.x, l.y), strokeWidth=wedgeStyle.label_pointer_strokeWidth, strokeColor=wedgeStyle.label_pointer_strokeColor))
                            L_add(l)
                elif text:
                    slices_popout = self.slices.popout
                    m = 0
                    for n, angle in angles:
                        if self.slices[n].fillColor:
                            m += 1
                        else:
                            r = n % m
                            self.slices[n].fillColor = self.slices[r].fillColor
                            self.slices[n].popout = self.slices[r].popout
                    for j in range(0, m - 1):
                        if self.slices[j].popout > slices_popout:
                            slices_popout = self.slices[j].popout
                    labelRadius = wedgeStyle.labelRadius
                    ry = yradius * labelRadius
                    if abs(averageAngle) < 90 or (averageAngle > 270 and averageAngle < 450) or -450 < averageAngle < -270:
                        labelX = (1 + self.sideLabelsOffset) * self.width + self.x + slices_popout
                        rx = 0
                    else:
                        labelX = self.x - self.sideLabelsOffset * self.width - slices_popout
                        rx = 0
                    labelY = cy + ry * sinAA
                    l = _addWedgeLabel(self, text, averageAngle, labelX, labelY, wedgeStyle)
                    L_add(l)
                    if not plMode:
                        l._aax = cx + xradius * cosAA
                        l._aay = cy + yradius * sinAA
                    if checkLabelOverlap:
                        l._origdata = {'x': labelX, 'y': labelY, 'angle': averageAngle, 'rx': rx, 'ry': ry, 'cx': cx, 'cy': cy, 'bounds': l.getBounds()}
                    x1, y1, x2, y2 = l.getBounds()
        if checkLabelOverlap and L:
            fixLabelOverlaps(L, sideLabels, mult0=checkLabelOverlap)
        for l in L:
            g_add(l)
        if not plMode:
            for l in L:
                if l._simple_pointer and (not sideLabels):
                    g_add(Line(l.x, l.y, l._aax, l._aay, strokeWidth=wedgeStyle.label_pointer_strokeWidth, strokeColor=wedgeStyle.label_pointer_strokeColor))
                elif sideLabels:
                    x1, y1, x2, y2 = l.getBounds()
                    if l.x == (1 + self.sideLabelsOffset) * self.width + self.x:
                        g_add(Line(l._aax, l._aay, 0.5 * (l._aax + l.x), l.y + 0.25 * (y2 - y1), strokeWidth=wedgeStyle.label_pointer_strokeWidth, strokeColor=wedgeStyle.label_pointer_strokeColor))
                        g_add(Line(0.5 * (l._aax + l.x), l.y + 0.25 * (y2 - y1), l.x, l.y + 0.25 * (y2 - y1), strokeWidth=wedgeStyle.label_pointer_strokeWidth, strokeColor=wedgeStyle.label_pointer_strokeColor))
                    else:
                        g_add(Line(l._aax, l._aay, 0.5 * (l._aax + l.x), l.y + 0.25 * (y2 - y1), strokeWidth=wedgeStyle.label_pointer_strokeWidth, strokeColor=wedgeStyle.label_pointer_strokeColor))
                        g_add(Line(0.5 * (l._aax + l.x), l.y + 0.25 * (y2 - y1), l.x, l.y + 0.25 * (y2 - y1), strokeWidth=wedgeStyle.label_pointer_strokeWidth, strokeColor=wedgeStyle.label_pointer_strokeColor))
        return g

    def draw(self):
        G = self.makeBackground()
        w = self.makeWedges()
        if G:
            return Group(G, w)
        return w