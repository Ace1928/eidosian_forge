from reportlab.lib.colors import black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.fonts import tt2ps
from reportlab.rl_config import canvas_basefontname as _baseFontName, \
class ListStyle(PropertySet):
    defaults = dict(leftIndent=18, rightIndent=0, bulletAlign='left', bulletType='1', bulletColor=black, bulletFontName='Helvetica', bulletFontSize=12, bulletOffsetY=0, bulletDedent='auto', bulletDir='ltr', bulletFormat=None, start=None)