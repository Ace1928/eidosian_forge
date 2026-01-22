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
class _Container(_ContainerSpace):

    def drawOn(self, canv, x, y, _sW=0, scale=1.0, content=None, aW=None):
        """we simulate being added to a frame"""
        from reportlab.platypus.doctemplate import ActionFlowable, Indenter
        x0 = x
        y0 = y
        pS = 0
        if aW is None:
            aW = self.width
        aW *= scale
        if content is None:
            content = self._content
        x = self._hAlignAdjust(x, _sW * scale)
        y += self.height * scale
        yt = y
        frame = getattr(self, '_frame', None)
        for c in content:
            if not ignoreContainerActions and isinstance(c, ActionFlowable):
                c.apply(canv._doctemplate)
                continue
            if isinstance(c, Indenter):
                x += c.left * scale
                aW -= (c.left + c.right) * scale
                continue
            w, h = c.wrapOn(canv, aW, 268435455)
            if h < _FUZZ and (not getattr(c, '_ZEROSIZE', None)):
                continue
            if yt != y:
                s = c.getSpaceBefore()
                if not getattr(c, '_SPACETRANSFER', False):
                    h += max(s - pS, 0)
            y -= h
            s = c.getSpaceAfter()
            if getattr(c, '_SPACETRANSFER', False):
                s = pS
            pS = s
            fbg = getattr(frame, '_frameBGs', None)
            if fbg and fbg[-1].active:
                bg = fbg[-1]
                fbgl = bg.left
                fbgr = bg.right
                bgm = bg.start
                fbw = scale * (frame._width - fbgl - fbgr)
                fbx = x0 + scale * (fbgl - frame._leftPadding)
                fbh = y + h + pS
                fby = max(y0, y - pS)
                fbh = max(0, fbh - fby)
                bg.render(canv, frame, fbx, fby, fbw, fbh)
            c._frame = frame
            c.drawOn(canv, x, y, _sW=aW - w)
            if c is not content[-1] and (not getattr(c, '_SPACETRANSFER', None)):
                y -= pS
            del c._frame

    def copyContent(self, content=None):
        C = [].append
        for c in content or self._content:
            C(cdeepcopy(c))
        self._content = C.__self__