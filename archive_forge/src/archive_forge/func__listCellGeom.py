from reportlab.platypus.flowables import Flowable, Preformatted
from reportlab import rl_config
from reportlab.lib.styles import PropertySet, ParagraphStyle, _baseFontName
from reportlab.lib import colors
from reportlab.lib.utils import annotateException, IdentStr, flatten, isStr, asNative, strTypes, __UNSET__
from reportlab.lib.validators import isListOfNumbersOrNone
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.abag import ABag as CellFrame
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus.doctemplate import Indenter, NullActionFlowable
from reportlab.platypus.flowables import LIIndenter
from collections import namedtuple
def _listCellGeom(self, V, w, s, W=None, H=None, aH=72000):
    if not V:
        return (0, 0)
    aW = w - s.leftPadding - s.rightPadding
    aH = aH - s.topPadding - s.bottomPadding
    t = 0
    w = 0
    canv = getattr(self, 'canv', None)
    sb0 = None
    if isinstance(V, str):
        vw = self._elementWidth(V, s)
        vh = len(V.split('\n')) * s.fontsize * 1.2
        return (max(w, vw), vh)
    for v in V:
        vw, vh = v.wrapOn(canv, aW, aH)
        sb = v.getSpaceBefore()
        sa = v.getSpaceAfter()
        if W is not None:
            W.append(vw)
        if H is not None:
            H.append(vh)
        w = max(w, vw)
        t += vh + sa + sb
        if sb0 is None:
            sb0 = sb
    return (w, t - sb0 - sa)