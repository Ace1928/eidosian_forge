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
def _calcBezierPoints(P, kind):
    """calculate all or half of a bezier curve
    kind==0 all, 1=first half else second half"""
    if kind == 0:
        return P
    else:
        Q0 = (0.5 * (P[0][0] + P[1][0]), 0.5 * (P[0][1] + P[1][1]))
        Q1 = (0.5 * (P[1][0] + P[2][0]), 0.5 * (P[1][1] + P[2][1]))
        Q2 = (0.5 * (P[2][0] + P[3][0]), 0.5 * (P[2][1] + P[3][1]))
        R0 = (0.5 * (Q0[0] + Q1[0]), 0.5 * (Q0[1] + Q1[1]))
        R1 = (0.5 * (Q1[0] + Q2[0]), 0.5 * (Q1[1] + Q2[1]))
        S0 = (0.5 * (R0[0] + R1[0]), 0.5 * (R0[1] + R1[1]))
        return [P[0], Q0, R0, S0] if kind == 1 else [S0, R1, Q2, P[3]]