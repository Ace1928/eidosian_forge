from reportlab.lib.units import inch,cm
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.formatters import DecimalFormatter
from reportlab.graphics.shapes import definePath, Group, Drawing, Rect, PolyLine, String
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.colors import Color, black, white, ReportLabBlue
from reportlab.pdfbase.pdfmetrics import stringWidth
class RL_CorpLogo(Widget):
    """Dinu's fat letter logo as hacked into decent paths by Robin"""
    _attrMap = AttrMap(x=AttrMapValue(isNumber, 'Logo x-coord'), y=AttrMapValue(isNumber, 'Logo y-coord'), angle=AttrMapValue(isNumber, 'Logo rotation'), strokeColor=AttrMapValue(isColorOrNone, 'Logo lettering stroke color'), fillColor=AttrMapValue(isColorOrNone, 'Logo lettering fill color'), strokeWidth=AttrMapValue(isNumber, 'Logo lettering stroke width'), background=AttrMapValue(isColorOrNone, desc='Logo background color'), border=AttrMapValue(isColorOrNone, desc='Logo border color'), borderWidth=AttrMapValue(isNumber, desc='Logo border width (1)'), shadow=AttrMapValue(isNumberOrNone, desc='None or fraction of background for shadowing'), width=AttrMapValue(isNumber, desc='width in points of the logo (default 129)'), height=AttrMapValue(isNumber, desc='height in points of the logo (default 86)'), skewX=AttrMapValue(isNumber, desc='x-skew of the logo (default 10)'), skewY=AttrMapValue(isNumber, desc='y-skew of the logo (default 0)'), showPage=AttrMapValue(EitherOr((isBoolean, SequenceOf(isBoolean, lo=2, hi=2))), desc='If true or (true(top),true(bottom)) show the page lines'), xFlip=AttrMapValue(isBoolean, desc='If true do x reversal'), yFlip=AttrMapValue(isBoolean, desc='If true do y reversal'), oColors=AttrMapValue(NoneOr(EitherOr((isColor, SequenceOf(isColorOrNone, lo=2, hi=2)))), desc='None or fill/stroke colors for the o in ReportLab'), pageColors=AttrMapValue(NoneOr(EitherOr((isColor, SequenceOf(isColorOrNone, lo=2, hi=2)))), desc='None or fill/stroke colors for the page outline'), prec=AttrMapValue(NoneOr(isInt), desc='None or precision negative means strip excess'))

    def __init__(self):
        self.fillColor = white
        self.strokeColor = None
        self.strokeWidth = 0.1
        self.background = ReportLabBlue
        self.border = None
        self.borderWidth = 1
        self.shadow = 0.5
        self.height = 86
        self.width = 130
        self.x = self.y = self.angle = self.skewY = self._dx = 0
        self.skewX = 10
        self._dy = 35.5
        self.showPage = 1
        self.oColors = None
        self.pageColors = None
        self.prec = None

    def demo(self):
        D = Drawing(self.width, self.height)
        D.add(self)
        return D

    def _paintLogo(self, g, dx=0, dy=0, strokeColor=None, strokeWidth=0.1, fillColor=white, _ocolors=None, _pagecolors=None):
        OP = [('moveTo', 62.10648, 6.51392), ('curveTo', 62.10648, 4.44205, 61.47118, 2.79288, 60.2003, 1.56631), ('curveTo', 58.92971, 0.33978, 57.22626, -0.27348, 55.08965, -0.27348), ('curveTo', 52.99018, -0.27348, 51.31914, 0.35221, 50.07595, 1.60362), ('curveTo', 48.8419, 2.8633, 48.22517, 4.55394, 48.22517, 6.67551), ('curveTo', 48.22517, 8.79709, 48.85575, 10.50016, 50.1175, 11.78472), ('curveTo', 51.36982, 13.07755, 53.03172, 13.72396, 55.1035, 13.72396), ('curveTo', 57.28608, 13.72396, 58.99866, 13.08168, 60.24185, 11.79712), ('curveTo', 61.48503, 10.51259, 62.10648, 8.75154, 62.10648, 6.51392), 'closePath', ('moveTo', 56.73358, 6.67551), ('curveTo', 56.73358, 7.17276, 56.69675, 7.62236, 56.62308, 8.02428), ('curveTo', 56.54942, 8.42623, 56.44334, 8.77016, 56.30544, 9.05607), ('curveTo', 56.16724, 9.34198, 56.00134, 9.56369, 55.80804, 9.72113), ('curveTo', 55.61474, 9.8786, 55.39817, 9.95733, 55.1589, 9.95733), ('curveTo', 54.68921, 9.95733, 54.31174, 9.65898, 54.02621, 9.06229), ('curveTo', 53.74068, 8.54018, 53.59807, 7.75702, 53.59807, 6.71282), ('curveTo', 53.59807, 5.68515, 53.74068, 4.90202, 54.02621, 4.36332), ('curveTo', 54.31174, 3.76663, 54.69392, 3.46828, 55.17275, 3.46828), ('curveTo', 55.62388, 3.46828, 55.99692, 3.7625, 56.29159, 4.35088), ('curveTo', 56.58625, 5.0056, 56.73358, 5.78047, 56.73358, 6.67551), 'closePath']
        P = [('moveTo', 15.7246, 0), ('lineTo', 9.49521, 0), ('lineTo', 6.64988, 6.83711), ('curveTo', 6.62224, 6.95315, 6.57391, 7.10646, 6.50485, 7.29708), ('curveTo', 6.43578, 7.48767, 6.35059, 7.71559, 6.24931, 7.98079), ('lineTo', 6.29074, 6.71282), ('lineTo', 6.29074, 0), ('lineTo', 0.55862, 0), ('lineTo', 0.55862, 19.19365), ('lineTo', 6.45649, 19.19365), ('curveTo', 9.05324, 19.19365, 10.99617, 18.73371, 12.28532, 17.8138), ('curveTo', 13.92439, 16.63697, 14.7439, 14.96293, 14.7439, 12.79161), ('curveTo', 14.7439, 10.47114, 13.64354, 8.86755, 11.44276, 7.98079), 'closePath', ('moveTo', 6.31838, 10.30542), ('lineTo', 6.70513, 10.30542), ('curveTo', 7.36812, 10.30542, 7.92062, 10.53331, 8.36261, 10.98912), ('curveTo', 8.80461, 11.44491, 9.0256, 12.02504, 9.0256, 12.72947), ('curveTo', 9.0256, 14.16321, 8.19227, 14.88004, 6.52556, 14.88004), ('lineTo', 6.31838, 14.88004), 'closePath', ('moveTo', 25.06173, 4.54978), ('lineTo', 30.47611, 4.45033), ('curveTo', 30.08951, 2.88402, 29.33668, 1.70513, 28.21787, 0.91369), ('curveTo', 27.09906, 0.12223, 25.63726, -0.27348, 23.83245, -0.27348), ('curveTo', 21.69611, -0.27348, 20.02024, 0.32322, 18.80475, 1.5166), ('curveTo', 17.59846, 2.72658, 16.99531, 4.37988, 16.99531, 6.47662), ('curveTo', 16.99531, 8.6065, 17.64451, 10.34269, 18.94286, 11.68527), ('curveTo', 20.24124, 13.03612, 21.91711, 13.71152, 23.97056, 13.71152), ('curveTo', 26.01482, 13.71152, 27.64466, 13.06096, 28.86015, 11.75985), ('curveTo', 30.07566, 10.45042, 30.68326, 8.71423, 30.68326, 6.5512), ('lineTo', 30.65586, 5.66859), ('lineTo', 22.53407, 5.66859), ('curveTo', 22.59855, 4.29287, 23.03132, 3.60503, 23.83245, 3.60503), ('curveTo', 24.45861, 3.60503, 24.86837, 3.91994, 25.06173, 4.54978), 'closePath', ('moveTo', 25.18604, 8.35371), ('curveTo', 25.18604, 8.60235, 25.15384, 8.83024, 25.08937, 9.03742), ('curveTo', 25.02489, 9.24463, 24.93514, 9.42278, 24.82001, 9.57197), ('curveTo', 24.70492, 9.72113, 24.56911, 9.83923, 24.41255, 9.92624), ('curveTo', 24.25603, 10.01326, 24.08568, 10.05678, 23.90152, 10.05678), ('curveTo', 23.51474, 10.05678, 23.20169, 9.89725, 22.96225, 9.57819), ('curveTo', 22.72283, 9.25913, 22.60314, 8.85096, 22.60314, 8.35371), 'closePath', ('moveTo', 38.36308, -5.99181), ('lineTo', 32.82428, -5.99181), ('lineTo', 32.82428, 13.43804), ('lineTo', 38.36308, 13.43804), ('lineTo', 38.23873, 11.53608), ('curveTo', 38.46886, 11.93387, 38.70371, 12.27159, 38.94327, 12.54922), ('curveTo', 39.18254, 12.82685, 39.44037, 13.05268, 39.71676, 13.22671), ('curveTo', 39.99286, 13.40074, 40.28988, 13.52712, 40.60753, 13.60585), ('curveTo', 40.92518, 13.68459, 41.27759, 13.72396, 41.66419, 13.72396), ('curveTo', 43.10068, 13.72396, 44.2702, 13.07755, 45.17246, 11.78472), ('curveTo', 46.06588, 10.50844, 46.51229, 8.81368, 46.51229, 6.70038), ('curveTo', 46.51229, 4.55394, 46.08415, 2.85502, 45.22785, 1.60362), ('curveTo', 44.38983, 0.35221, 43.23416, -0.27348, 41.76084, -0.27348), ('curveTo', 40.41659, -0.27348, 39.24235, 0.42679, 38.23873, 1.82739), ('curveTo', 38.2847, 1.40472, 38.31239, 1.04007, 38.32153, 0.73345), ('curveTo', 38.34923, 0.41851, 38.36308, 0.04146, 38.36308, -0.3978), 'closePath', ('moveTo', 40.7802, 6.84954), ('curveTo', 40.7802, 7.72802, 40.66734, 8.40964, 40.44193, 8.89448), ('curveTo', 40.21621, 9.37929, 39.89621, 9.62168, 39.48191, 9.62168), ('curveTo', 38.62533, 9.62168, 38.19718, 8.68108, 38.19718, 6.79983), ('curveTo', 38.19718, 4.87712, 38.61177, 3.91581, 39.44037, 3.91581), ('curveTo', 39.85466, 3.91581, 40.18174, 4.1727, 40.42101, 4.68654), ('curveTo', 40.66057, 5.20037, 40.7802, 5.92135, 40.7802, 6.84954), 'closePath', ('moveTo', 69.78629, 0), ('lineTo', 64.2475, 0), ('lineTo', 64.2475, 13.43804), ('lineTo', 69.78629, 13.43804), ('lineTo', 69.49605, 10.81507), ('curveTo', 70.33407, 12.77921, 71.71988, 13.76126, 73.65346, 13.76126), ('lineTo', 73.65346, 8.16725), ('curveTo', 73.04586, 8.4656, 72.5302, 8.61478, 72.10647, 8.61478), ('curveTo', 71.36068, 8.61478, 70.78756, 8.37236, 70.38711, 7.88755), ('curveTo', 69.98637, 7.40274, 69.78629, 6.69623, 69.78629, 5.76804), 'closePath', ('moveTo', 81.55427, 0), ('lineTo', 76.00163, 0), ('lineTo', 76.00163, 9.42278), ('lineTo', 74.42725, 9.42278), ('lineTo', 74.42725, 13.43804), ('lineTo', 76.00163, 13.43804), ('lineTo', 76.00163, 17.39113), ('lineTo', 81.55427, 17.39113), ('lineTo', 81.55427, 13.43804), ('lineTo', 83.39121, 13.43804), ('lineTo', 83.39121, 9.42278), ('lineTo', 81.55427, 9.42278), 'closePath', ('moveTo', 95.17333, 0), ('lineTo', 85.09024, 0), ('lineTo', 85.09024, 19.19365), ('lineTo', 90.85002, 19.19365), ('lineTo', 90.85002, 4.61196), ('lineTo', 95.17333, 4.61196), 'closePath', ('moveTo', 110.00787, 0), ('lineTo', 104.45523, 0), ('curveTo', 104.5012, 0.44754, 104.53803, 0.87433, 104.56573, 1.2804), ('curveTo', 104.59313, 1.68651, 104.62083, 2.01385, 104.64853, 2.26246), ('curveTo', 103.69087, 0.57182, 102.40644, -0.27348, 100.79492, -0.27348), ('curveTo', 99.39527, -0.27348, 98.28557, 0.35637, 97.46611, 1.61605), ('curveTo', 96.65578, 2.86746, 96.25062, 4.59952, 96.25062, 6.81227), ('curveTo', 96.25062, 8.95041, 96.66963, 10.63276, 97.50765, 11.8593), ('curveTo', 98.34538, 13.10242, 99.4872, 13.72396, 100.93312, 13.72396), ('curveTo', 102.41557, 13.72396, 103.61249, 12.92008, 104.52418, 11.31231), ('curveTo', 104.50591, 11.47806, 104.49206, 11.62309, 104.48293, 11.74741), ('curveTo', 104.4735, 11.87173, 104.46437, 11.9753, 104.45523, 12.05819), ('lineTo', 104.39983, 12.84135), ('lineTo', 104.35858, 13.43804), ('lineTo', 110.00787, 13.43804), 'closePath', ('moveTo', 104.39983, 6.88685), ('curveTo', 104.39983, 7.38409, 104.37921, 7.80676, 104.33766, 8.15481), ('curveTo', 104.29641, 8.5029, 104.22952, 8.78672, 104.13758, 9.00636), ('curveTo', 104.04535, 9.22598, 103.92572, 9.38341, 103.77839, 9.47874), ('curveTo', 103.63106, 9.57403, 103.45161, 9.62168, 103.23974, 9.62168), ('curveTo', 102.30036, 9.62168, 101.83096, 8.49875, 101.83096, 6.25285), ('curveTo', 101.83096, 4.64508, 102.24967, 3.8412, 103.0877, 3.8412), ('curveTo', 103.96255, 3.8412, 104.39983, 4.85641, 104.39983, 6.88685), 'closePath', ('moveTo', 118.22604, 0), ('lineTo', 112.5629, 0), ('lineTo', 112.5629, 20.99616), ('lineTo', 118.10169, 20.99616), ('lineTo', 118.10169, 13.63694), ('curveTo', 118.10169, 13.01538, 118.07399, 12.30268, 118.01889, 11.49877), ('curveTo', 118.52542, 12.31096, 119.03636, 12.88693, 119.55202, 13.22671), ('curveTo', 120.08625, 13.55821, 120.75838, 13.72396, 121.5687, 13.72396), ('curveTo', 123.07885, 13.72396, 124.24837, 13.09827, 125.07697, 11.84686), ('curveTo', 125.90586, 10.60373, 126.32015, 8.85099, 126.32015, 6.5885), ('curveTo', 126.32015, 4.42546, 125.89201, 2.74314, 125.03571, 1.54147), ('curveTo', 124.18826, 0.3315, 123.01432, -0.27348, 121.51331, -0.27348), ('curveTo', 120.78608, -0.27348, 120.16905, -0.12432, 119.66252, 0.17403), ('curveTo', 119.41383, 0.3315, 119.15835, 0.54283, 118.8961, 0.80803), ('curveTo', 118.63356, 1.07322, 118.36866, 1.40472, 118.10169, 1.80252), ('curveTo', 118.11112, 1.64505, 118.12025, 1.51039, 118.12939, 1.3985), ('curveTo', 118.13852, 1.28662, 118.14766, 1.19339, 118.15709, 1.11881), 'closePath', ('moveTo', 120.58806, 6.70038), ('curveTo', 120.58806, 8.62306, 120.11837, 9.5844, 119.17898, 9.5844), ('curveTo', 118.35039, 9.5844, 117.93609, 8.67693, 117.93609, 6.86198), ('curveTo', 117.93609, 4.96417, 118.36424, 4.01526, 119.22053, 4.01526), ('curveTo', 120.13222, 4.01526, 120.58806, 4.91027, 120.58806, 6.70038), 'closePath']
        showPage = self.showPage
        PP = []
        if showPage:
            if not isSeq(showPage):
                showPage = (showPage, showPage)
            if showPage[0]:
                PP.extend([('moveTo', 32.70766, 14.52164), ('lineTo', 32.70766, 47.81862), ('lineTo', 80.14849, 47.81862), ('lineTo', 90.46172, 37.21073), ('lineTo', 90.46172, 20.12025), ('lineTo', 85.15777, 20.12025), ('lineTo', 85.15777, 30.72814), ('lineTo', 73.66589, 30.72814), ('lineTo', 73.66589, 42.22002), ('lineTo', 38.30626, 42.22002), ('lineTo', 38.30626, 14.52164), 'closePath', ('moveTo', 79.2645, 36.32674), ('lineTo', 85.15777, 36.32674), ('lineTo', 79.2645, 42.22002), 'closePath'])
            if showPage[1]:
                PP.extend([('moveTo', 38.30626, -7.28346), ('lineTo', 38.30626, -25.55261), ('lineTo', 85.15777, -25.55261), ('lineTo', 85.15777, -1.39019), ('lineTo', 90.46172, -1.39019), ('lineTo', 90.46172, -31.15121), ('lineTo', 32.70766, -31.15121), ('lineTo', 32.70766, -7.28346), 'closePath'])
        if _ocolors:
            OP = self.applyPrec(OP, self.prec)
            g.add(definePath(OP, strokeColor=_ocolors[0], strokeWidth=strokeWidth, fillColor=_ocolors[1], dx=dx, dy=dy))
        else:
            P += OP
        if self.showPage and _pagecolors:
            PP = self.applyPrec(PP, self.prec)
            g.add(definePath(PP, strokeColor=_pagecolors[0], strokeWidth=strokeWidth, fillColor=_pagecolors[1], dx=dx, dy=dy))
        else:
            P += PP
        P = self.applyPrec(P, self.prec)
        g.add(definePath(P, strokeColor=strokeColor, strokeWidth=strokeWidth, fillColor=fillColor, dx=dx, dy=dy))

    @staticmethod
    def applyPrec(P, prec):
        if prec is None:
            return P
        R = [].append
        f = DecimalFormatter(places=prec)
        for p in P:
            if isSeq(p):
                n = [].append
                for e in p:
                    if isinstance(e, float):
                        e = float(f(e))
                    n(e)
                p = n.__self__
            R(p)
        return R.__self__

    def draw(self):
        fillColor = self.fillColor
        strokeColor = self.strokeColor
        g = Group()
        bg = self.background
        bd = self.border
        bdw = self.borderWidth
        shadow = self.shadow
        x, y = (self.x, self.y)
        if bg:
            if shadow is not None and 0 <= shadow < 1:
                shadow = Color(bg.red * shadow, bg.green * shadow, bg.blue * shadow)
            self._paintLogo(g, dy=-2.5, dx=2, fillColor=shadow)
        self._paintLogo(g, fillColor=fillColor, strokeColor=strokeColor, _ocolors=self.oColors or None, _pagecolors=self.pageColors or None)
        g.skew(kx=self.skewX, ky=self.skewY)
        g.shift(self._dx, self._dy)
        G = Group()
        G.add(g)
        _w, _h = (130, 86)
        w, h = (self.width, self.height)
        if bg or (bd and bdw):
            G.insert(0, Rect(0, 0, _w, _h, fillColor=bg, strokeColor=bd, strokeWidth=bdw))
        if w != _w or h != _h:
            G.scale(w / float(_w), h / float(_h))
        angle = self.angle
        if self.angle:
            w, h = (w / 2.0, h / 2.0)
            G.shift(-w, -h)
            G.rotate(angle)
            G.shift(w, h)
        xFlip = getattr(self, 'xFlip', 0) and -1 or 0
        yFlip = getattr(self, 'yFlip', 0) and -1 or 0
        if xFlip or yFlip:
            sx = xFlip or 1
            sy = yFlip or 1
            G.shift(sx * x + w * xFlip, sy * y + yFlip * h)
            G = Group(G, transform=(sx, 0, 0, sy, 0, 0))
        else:
            G.shift(x, y)
        return G