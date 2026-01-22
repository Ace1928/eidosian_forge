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
def _calc_height(self, availHeight, availWidth, H=None, W=None):
    H = self._argH
    if not W:
        W = _calc_pc(self._argW, availWidth)
    hmax = lim = len(H)
    longTable = self._longTableOptimize
    if None in H:
        minRowHeights = self._minRowHeights
        canv = getattr(self, 'canv', None)
        saved = None
        if self._spanCmds:
            rowSpanCells = self._rowSpanCells
            colSpanCells = self._colSpanCells
            spanRanges = self._spanRanges
            colpositions = self._colpositions
        else:
            rowSpanCells = colSpanCells = ()
            spanRanges = {}
        if canv:
            saved = (canv._fontname, canv._fontsize, canv._leading)
        H0 = H
        H = H[:]
        self._rowHeights = H
        spanCons = {}
        FUZZ = rl_config._FUZZ
        while None in H:
            i = H.index(None)
            V = self._cellvalues[i]
            S = self._cellStyles[i]
            h = 0
            j = 0
            for j, (v, s, w) in enumerate(list(zip(V, S, W))):
                ji = (j, i)
                span = spanRanges.get(ji, None)
                if ji in rowSpanCells and (not span):
                    continue
                else:
                    if isinstance(v, (tuple, list, Flowable)):
                        v = V[j] = self._cellListProcess(v, w, None)
                        if w is None and (not self._canGetWidth(v)):
                            raise ValueError(f"Flowable {v[0].identity()} in cell({i},{j}) can't have auto width\n{self.identity(30)}")
                        if canv:
                            canv._fontname, canv._fontsize, canv._leading = (s.fontname, s.fontsize, s.leading or 1.2 * s.fontsize)
                        if ji in colSpanCells:
                            if not span:
                                continue
                            w = max(colpositions[span[2] + 1] - colpositions[span[0]], w or 0)
                        dW, t = self._listCellGeom(v, w or self._listValueWidth(v), s)
                        if canv:
                            canv._fontname, canv._fontsize, canv._leading = saved
                        dW = dW + s.leftPadding + s.rightPadding
                        if not rl_config.allowTableBoundsErrors and dW > w:
                            from reportlab.platypus.doctemplate import LayoutError
                            raise LayoutError('Flowable %s (%sx%s points) too wide for cell(%d,%d) (%sx* points) in\n%s' % (v[0].identity(30), fp_str(dW), fp_str(t), i, j, fp_str(w), self.identity(30)))
                    else:
                        v = (v is not None and str(v) or '').split('\n')
                        t = (s.leading or 1.2 * s.fontsize) * len(v)
                    t += s.bottomPadding + s.topPadding
                    if span:
                        r0 = span[1]
                        r1 = span[3]
                        if r0 != r1:
                            x = (r0, r1)
                            spanCons[x] = max(spanCons.get(x, t), t)
                            t = 0
                if t > h:
                    h = t
            H[i] = max(minRowHeights[i], h) if minRowHeights else h
            if longTable:
                hmax = i + 1
                height = sum(H[:hmax])
                if height > availHeight:
                    if spanCons:
                        msr = max((x[1] for x in spanCons.keys()))
                        if hmax > msr:
                            break
        if None not in H:
            hmax = lim
        if spanCons:
            try:
                spanFixDim(H0, H, spanCons)
            except:
                annotateException('\nspanning problem in %s hmax=%s lim=%s avail=%s x %s\nH0=%r H=%r\nspanCons=%r' % (self.identity(), hmax, lim, availWidth, availHeight, H0, H, spanCons))
    self._rowpositions = j = []
    height = c = 0
    for i in range(hmax - 1, -1, -1):
        j.append(height)
        y = H[i] - c
        t = height + y
        c = t - height - y
        height = t
    j.append(height)
    self._height = height
    j.reverse()
    self._hmax = hmax