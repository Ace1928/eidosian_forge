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
def _splitLineCmds(self, n, doInRowSplit=0):
    nrows = self._nrows
    ncols = self._ncols
    A = []
    for op, (sc, sr), (ec, er), weight, color, cap, dash, join, count, space in self._linecmds:
        if isinstance(sr, strTypes) and sr in _SPECIALROWS:
            A.append((op, (sc, sr), (ec, sr), weight, color, cap, dash, join, count, space))
            if sr == 'splitlast':
                sr = er = n - 1
            elif sr == 'splitfirst':
                sr = n
                er = n
            else:
                if sc < 0:
                    sc += ncols
                if ec < 0:
                    ec += ncols
                A[-1] = (op, (sc, sr), (ec, sr), weight, color, cap, dash, join, count, space)
                continue
        if sc < 0:
            sc += ncols
        if ec < 0:
            ec += ncols
        if sr < 0:
            sr += nrows
        if er < 0:
            er += nrows
        if op in ('BOX', 'OUTLINE', 'GRID'):
            if sr < n and er >= n or (doInRowSplit and sr == n):
                A.append(('LINEABOVE', (sc, sr), (ec, sr), weight, color, cap, dash, join, count, space))
                A.append(('LINEBEFORE', (sc, sr), (sc, er), weight, color, cap, dash, join, count, space))
                A.append(('LINEAFTER', (ec, sr), (ec, er), weight, color, cap, dash, join, count, space))
                A.append(('LINEBELOW', (sc, er), (ec, er), weight, color, cap, dash, join, count, space))
                if op == 'GRID':
                    if doInRowSplit:
                        A.append(('INNERGRID', (sc, sr), (ec, n - 1), weight, color, cap, dash, join, count, space))
                        A.append(('INNERGRID', (sc, n), (ec, er), weight, color, cap, dash, join, count, space))
                    else:
                        A.append(('LINEBELOW', (sc, n - 1), (ec, n - 1), weight, color, cap, dash, join, count, space))
                        A.append(('LINEABOVE', (sc, n), (ec, n), weight, color, cap, dash, join, count, space))
                        A.append(('INNERGRID', (sc, sr), (ec, er), weight, color, cap, dash, join, count, space))
            else:
                A.append((op, (sc, sr), (ec, er), weight, color, cap, dash, join, count, space))
        elif op == 'INNERGRID':
            if sr < n and er >= n and (not doInRowSplit):
                A.append(('LINEBELOW', (sc, n - 1), (ec, n - 1), weight, color, cap, dash, join, count, space))
                A.append(('LINEABOVE', (sc, n), (ec, n), weight, color, cap, dash, join, count, space))
            A.append((op, (sc, sr), (ec, er), weight, color, cap, dash, join, count, space))
        elif op == 'LINEBELOW':
            if sr < n and er >= n - 1:
                A.append(('LINEABOVE', (sc, n), (ec, n), weight, color, cap, dash, join, count, space))
            A.append((op, (sc, sr), (ec, er), weight, color, cap, dash, join, count, space))
        elif op == 'LINEABOVE':
            if sr <= n and er >= n:
                A.append(('LINEBELOW', (sc, n - 1), (ec, n - 1), weight, color, cap, dash, join, count, space))
            A.append((op, (sc, sr), (ec, er), weight, color, cap, dash, join, count, space))
        else:
            A.append((op, (sc, sr), (ec, er), weight, color, cap, dash, join, count, space))
    return A