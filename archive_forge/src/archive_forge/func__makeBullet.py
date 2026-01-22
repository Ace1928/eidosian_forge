import os
from copy import deepcopy, copy
from reportlab.lib.colors import gray, lightgrey
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import _baseFontName
from reportlab.lib.utils import strTypes, rl_safe_exec, annotateException
from reportlab.lib.abag import ABag
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ, overlapAttachedSpace, ignoreContainerActions, listWrapOnFakeWidth
from reportlab.lib.sequencer import _type2formatter
from reportlab.lib.styles import ListStyle
def _makeBullet(self, value, params=None):
    if params is None:

        def getp(a):
            return getattr(self, '_' + a)
    else:
        style = getattr(params, 'style', None)

        def getp(a):
            if a in params:
                return params[a]
            if style and a in style.__dict__:
                return getattr(self, a)
            return getattr(self, '_' + a)
    return BulletDrawer(value=value, bulletAlign=getp('bulletAlign'), bulletType=getp('bulletType'), bulletColor=getp('bulletColor'), bulletFontName=getp('bulletFontName'), bulletFontSize=getp('bulletFontSize'), bulletOffsetY=getp('bulletOffsetY'), bulletDedent=getp('calcBulletDedent'), bulletDir=getp('bulletDir'), bulletFormat=getp('bulletFormat'))