from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
class InternalLink(HotLink):

    def link(self, rect, canvas):
        destinationname = self.url
        contents = ''
        canvas.linkRect(contents, destinationname, rect, Border='[0 0 0]')