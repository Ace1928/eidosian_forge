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
class _FBGBag(ABag):

    def matches(self, frame, canv):
        fid = id(frame)
        return (isinstance(self.fid, list) and fid in self.fid or fid == self.fid) and id(canv) == self.cid and (self.pn == canv.getPageNumber())

    def getDims(self, canv):
        self._inst = canv._code[self.codePos].split()
        return map(float, self._inst[1:5])

    def setDims(self, canv, x, y, w, h):
        self._inst[1:5] = [fp_str(x, y, w, h)]
        canv._code[self.codePos] = ' '.join(self._inst)

    def render(self, canv, frame, fbx, fby, fbw, fbh):
        if abs(fbw) > _FUZZ and abs(fbh) > _FUZZ:
            pn = canv.getPageNumber()
            if self.fid == id(frame) and self.cid == id(canv) and (self.pn == pn):
                ox, oy, ow, oh = self.getDims(canv)
                self.setDims(canv, ox, fby, ow, oh + oy - fby)
            else:
                canv.saveState()
                fbgc = self.fillColor
                if fbgc:
                    canv.setFillColor(fbgc)
                sw = self.strokeWidth
                sc = None if sw is None or sw < 0 else self.strokeColor
                if sc:
                    canv.setStrokeColor(sc)
                    canv.setLineWidth(sw)
                    da = self.strokeDashArray
                    if da:
                        canv.setDash(da)
                self.fid = id(frame)
                self.cid = id(canv)
                self.pn = pn
                self.codePos = len(canv._code)
                canv.rect(fbx, fby, fbw, fbh, stroke=1 if sc else 0, fill=1 if fbgc else 0)
                canv.restoreState()