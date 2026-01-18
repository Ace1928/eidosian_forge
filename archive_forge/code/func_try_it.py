from reportlab.lib import PyFontify
from reportlab.platypus.paragraph import Paragraph, _handleBulletWidth, \
from reportlab.lib.utils import isSeq
from reportlab.platypus.flowables import _dedenter
def try_it(text, style, dedent, aW, aH):
    P = XPreformatted(text, style, dedent=dedent)
    dumpXPreformattedFrags(P)
    w, h = P.wrap(aW, aH)
    dumpXPreformattedLines(P)
    S = P.split(aW, aH)
    dumpXPreformattedLines(P)
    for s in S:
        s.wrap(aW, aH)
        dumpXPreformattedLines(s)
        aH = 500