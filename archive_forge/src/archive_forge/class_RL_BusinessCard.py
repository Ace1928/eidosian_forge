from reportlab.lib.units import inch,cm
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.formatters import DecimalFormatter
from reportlab.graphics.shapes import definePath, Group, Drawing, Rect, PolyLine, String
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.colors import Color, black, white, ReportLabBlue
from reportlab.pdfbase.pdfmetrics import stringWidth
class RL_BusinessCard(Widget):
    """Widget that creates a single business card.
    Uses RL_CorpLogo for the logo.

    For a black border around your card, set self.border to 1.
    To change the details on the card, over-ride the following properties:
    self.name, self.position, self.telephone, self.mobile, self.fax, self.email, self.web
    The office locations are set in self.rh_blurb_top ("London office" etc), and
    self.rh_blurb_bottom ("New York office" etc).
    """
    _attrMap = AttrMap(fillColor=AttrMapValue(isColorOrNone), strokeColor=AttrMapValue(isColorOrNone), altStrokeColor=AttrMapValue(isColorOrNone), x=AttrMapValue(isNumber), y=AttrMapValue(isNumber), height=AttrMapValue(isNumber), width=AttrMapValue(isNumber), borderWidth=AttrMapValue(isNumber), bleed=AttrMapValue(isNumberOrNone), cropMarks=AttrMapValue(isBoolean), border=AttrMapValue(isBoolean), name=AttrMapValue(isString), position=AttrMapValue(isString), telephone=AttrMapValue(isString), mobile=AttrMapValue(isString), fax=AttrMapValue(isString), email=AttrMapValue(isString), web=AttrMapValue(isString), rh_blurb_top=AttrMapValue(isListOfStringsOrNone), rh_blurb_bottom=AttrMapValue(isListOfStringsOrNone))
    _h = 5.35 * cm
    _w = 8.5 * cm
    _fontName = 'Helvetica-Bold'
    _strapline = 'strategic reporting solutions for e-business'

    def __init__(self):
        self.fillColor = ReportLabBlue
        self.strokeColor = black
        self.altStrokeColor = white
        self.x = 0
        self.y = 0
        self.height = self._h
        self.width = self._w
        self.borderWidth = self.width / 6.15
        self.bleed = 0.2 * cm
        self.cropMarks = 1
        self.border = 0
        self.name = 'Joe Cool'
        self.position = 'Freelance Demonstrator'
        self.telephone = '020 8545 7271'
        self.mobile = '-'
        self.fax = '020 8544 1311'
        self.email = 'info@reportlab.com'
        self.web = 'www.reportlab.com'
        self.rh_blurb_top = ['London office:', 'ReportLab Europe Ltd', 'Media House', '3 Palmerston Road', 'Wimbledon', 'London SW19 1PG', 'United Kingdom']

    def demo(self):
        D = Drawing(self.width, self.height)
        D.add(self)
        return D

    def draw(self):
        fillColor = self.fillColor
        strokeColor = self.strokeColor
        g = Group()
        g.add(Rect(x=0, y=0, fillColor=self.fillColor, strokeColor=self.fillColor, width=self.borderWidth, height=self.height))
        g.add(Rect(x=0, y=self.height - self.borderWidth, fillColor=self.fillColor, strokeColor=self.fillColor, width=self.width, height=self.borderWidth))
        g2 = Group()
        rl = RL_CorpLogo()
        rl.height = 1.25 * cm
        rl.width = 1.9 * cm
        rl.draw()
        g2.add(rl)
        g.add(g2)
        g2.shift(x=self.width - (rl.width + self.width / 42), y=self.height - (rl.height + self.height / 42))
        g.add(String(x=self.borderWidth / 5.0, y=self.height - (rl.height + self.height / 42) + 38 / 90.5 * rl.height, fontSize=6, fillColor=self.altStrokeColor, fontName='Helvetica-BoldOblique', textAnchor='start', text=self._strapline))
        leftText = ['Tel:', 'Mobile:', 'Fax:', 'Email:', 'Web:']
        leftDetails = [self.telephone, self.mobile, self.fax, self.email, self.web]
        leftText.reverse()
        leftDetails.reverse()
        for f in range(len(leftText), 0, -1):
            g.add(String(x=self.borderWidth + self.borderWidth / 5.0, y=self.borderWidth / 5.0 + (f - 1) * (5 * 1.2), fontSize=5, fillColor=self.strokeColor, fontName='Helvetica', textAnchor='start', text=leftText[f - 1]))
            g.add(String(x=self.borderWidth + self.borderWidth / 5.0 + self.borderWidth, y=self.borderWidth / 5.0 + (f - 1) * (5 * 1.2), fontSize=5, fillColor=self.strokeColor, fontName='Helvetica', textAnchor='start', text=leftDetails[f - 1]))
        ty = self.height - self.borderWidth - self.borderWidth / 5.0 + 2
        rightText = self.rh_blurb_top
        for f in range(1, len(rightText) + 1):
            g.add(String(x=self.width - self.borderWidth / 5.0, y=ty - f * (5 * 1.2), fontSize=5, fillColor=self.strokeColor, fontName='Helvetica', textAnchor='end', text=rightText[f - 1]))
        g.add(String(x=self.borderWidth + self.borderWidth / 5.0, y=ty - 10, fontSize=10, fillColor=self.strokeColor, fontName='Helvetica', textAnchor='start', text=self.name))
        ty1 = ty - 10 * 1.2
        g.add(String(x=self.borderWidth + self.borderWidth / 5.0, y=ty1 - 8, fontSize=8, fillColor=self.strokeColor, fontName='Helvetica', textAnchor='start', text=self.position))
        if self.border:
            g.add(Rect(x=0, y=0, fillColor=None, strokeColor=black, width=self.width, height=self.height))
        g.shift(self.x, self.y)
        return g