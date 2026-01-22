from reportlab.lib import PyFontify
from reportlab.platypus.paragraph import Paragraph, _handleBulletWidth, \
from reportlab.lib.utils import isSeq
from reportlab.platypus.flowables import _dedenter
def _split_blPara(blPara, start, stop):
    f = blPara.clone()
    for a in ('lines', 'text'):
        if hasattr(f, a):
            delattr(f, a)
    f.lines = blPara.lines[start:stop]
    return [f]