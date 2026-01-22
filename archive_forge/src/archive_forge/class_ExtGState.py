import re
import hashlib
from string import digits
from math import sin, cos, tan, pi
from reportlab import rl_config
from reportlab.pdfbase import pdfdoc
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen  import pathobject
from reportlab.pdfgen.textobject import PDFTextObject, _PDFColorSetter
from reportlab.lib.colors import black, _chooseEnforceColorSpace, Color, CMYKColor, toColor
from reportlab.lib.utils import ImageReader, isSeq, isStr, isUnicode, _digester, asUnicode
from reportlab.lib.rl_accel import fp_str, escapePDF
from reportlab.lib.boxstuff import aspectRatioFix
class ExtGState:
    defaults = dict(CA=1, ca=1, OP=False, op=False, OPM=0, BM='Normal')
    allowed = dict(BM={'Normal', 'Multiply', 'Screen', 'Overlay', 'Darken', 'Lighten', 'ColorDodge', 'ColorBurn', 'HardLight', 'SoftLight', 'Difference', 'Exclusion', 'Hue', 'Saturation', 'Color', 'Luminosity'})
    pdfNameValues = {'BM'}

    @staticmethod
    def _boolTransform(v):
        return str(v).lower()

    @staticmethod
    def _identityTransform(v):
        return v

    @staticmethod
    def _pdfNameTransform(v):
        return '/' + v

    def __init__(self):
        self._d = {}
        self._c = {}

    def set(self, canv, a, v):
        d = self.defaults[a]
        if isinstance(d, bool):
            v = bool(v)
            vTransform = self._boolTransform
        elif a in self.pdfNameValues:
            if v not in self.allowed[a]:
                raise ValueError('ExtGstate[%r] = %r not in allowed values %r' % (a, v, self.allowed[a]))
            vTransform = self._pdfNameTransform
        else:
            vTransform = self._identityTransform
        if v != self._d.get(a, d) or (a == 'op' and self.getValue('OP') != d):
            self._d[a] = v
            t = (a, vTransform(v))
            if t in self._c:
                name = self._c[t]
            else:
                name = 'gRLs' + str(len(self._c))
                self._c[t] = name
            canv._code.append('/%s gs' % name)

    def getValue(self, a):
        return self._d.get(a, self.defaults[a])

    def getState(self):
        S = {}
        for t, name in self._c.items():
            S[name] = pdfdoc.PDFDictionary(dict((t,)))
        return S and pdfdoc.PDFDictionary(S) or None

    def pushCopy(self):
        """the states must be shared across push/pop, but the values not"""
        x = self.__class__()
        x._d = self._d.copy()
        x._c = self._c
        return x