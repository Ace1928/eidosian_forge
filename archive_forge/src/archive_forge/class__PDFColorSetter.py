from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
class _PDFColorSetter:
    """Abstracts the color setting operations; used in Canvas and Textobject
    asseumes we have a _code object"""

    def _checkSeparation(self, cmyk):
        if isinstance(cmyk, CMYKColorSep):
            name, sname = self._doc.addColor(cmyk)
            if name not in self._colorsUsed:
                self._colorsUsed[name] = sname
            return name
    _enforceColorSpace = None

    def setFillColorCMYK(self, c, m, y, k, alpha=None):
        """set the fill color useing negative color values
         (cyan, magenta, yellow and darkness value).
         Takes 4 arguments between 0.0 and 1.0"""
        self.setFillColor((c, m, y, k), alpha=alpha)

    def setStrokeColorCMYK(self, c, m, y, k, alpha=None):
        """set the stroke color useing negative color values
            (cyan, magenta, yellow and darkness value).
            Takes 4 arguments between 0.0 and 1.0"""
        self.setStrokeColor((c, m, y, k), alpha=alpha)

    def setFillColorRGB(self, r, g, b, alpha=None):
        """Set the fill color using positive color description
           (Red,Green,Blue).  Takes 3 arguments between 0.0 and 1.0"""
        self.setFillColor((r, g, b), alpha=alpha)

    def setStrokeColorRGB(self, r, g, b, alpha=None):
        """Set the stroke color using positive color description
           (Red,Green,Blue).  Takes 3 arguments between 0.0 and 1.0"""
        self.setStrokeColor((r, g, b), alpha=alpha)

    def setFillColor(self, aColor, alpha=None):
        """Takes a color object, allowing colors to be referred to by name"""
        if self._enforceColorSpace:
            aColor = self._enforceColorSpace(aColor)
        if isinstance(aColor, CMYKColor):
            d = aColor.density
            c, m, y, k = (d * aColor.cyan, d * aColor.magenta, d * aColor.yellow, d * aColor.black)
            self._fillColorObj = aColor
            name = self._checkSeparation(aColor)
            if name:
                self._code.append('/%s cs %s scn' % (name, fp_str(d)))
            else:
                self._code.append('%s k' % fp_str(c, m, y, k))
        elif isinstance(aColor, Color):
            rgb = (aColor.red, aColor.green, aColor.blue)
            self._fillColorObj = aColor
            self._code.append('%s rg' % fp_str(rgb))
        elif isinstance(aColor, (tuple, list)):
            l = len(aColor)
            if l == 3:
                self._fillColorObj = aColor
                self._code.append('%s rg' % fp_str(aColor))
            elif l == 4:
                self._fillColorObj = aColor
                self._code.append('%s k' % fp_str(aColor))
            else:
                raise ValueError('Unknown color %r' % aColor)
        elif isStr(aColor):
            self.setFillColor(toColor(aColor))
        else:
            raise ValueError('Unknown color %r' % aColor)
        if alpha is not None:
            self.setFillAlpha(alpha)
        elif getattr(aColor, 'alpha', None) is not None:
            self.setFillAlpha(aColor.alpha)

    def setStrokeColor(self, aColor, alpha=None):
        """Takes a color object, allowing colors to be referred to by name"""
        if self._enforceColorSpace:
            aColor = self._enforceColorSpace(aColor)
        if isinstance(aColor, CMYKColor):
            d = aColor.density
            c, m, y, k = (d * aColor.cyan, d * aColor.magenta, d * aColor.yellow, d * aColor.black)
            self._strokeColorObj = aColor
            name = self._checkSeparation(aColor)
            if name:
                self._code.append('/%s CS %s SCN' % (name, fp_str(d)))
            else:
                self._code.append('%s K' % fp_str(c, m, y, k))
        elif isinstance(aColor, Color):
            rgb = (aColor.red, aColor.green, aColor.blue)
            self._strokeColorObj = aColor
            self._code.append('%s RG' % fp_str(rgb))
        elif isinstance(aColor, (tuple, list)):
            l = len(aColor)
            if l == 3:
                self._strokeColorObj = aColor
                self._code.append('%s RG' % fp_str(aColor))
            elif l == 4:
                self._strokeColorObj = aColor
                self._code.append('%s K' % fp_str(aColor))
            else:
                raise ValueError('Unknown color %r' % aColor)
        elif isStr(aColor):
            self.setStrokeColor(toColor(aColor))
        else:
            raise ValueError('Unknown color %r' % aColor)
        if alpha is not None:
            self.setStrokeAlpha(alpha)
        elif getattr(aColor, 'alpha', None) is not None:
            self.setStrokeAlpha(aColor.alpha)

    def setFillGray(self, gray, alpha=None):
        """Sets the gray level; 0.0=black, 1.0=white"""
        self._fillColorObj = (gray, gray, gray)
        self._code.append('%s g' % fp_str(gray))
        if alpha is not None:
            self.setFillAlpha(alpha)

    def setStrokeGray(self, gray, alpha=None):
        """Sets the gray level; 0.0=black, 1.0=white"""
        self._strokeColorObj = (gray, gray, gray)
        self._code.append('%s G' % fp_str(gray))
        if alpha is not None:
            self.setFillAlpha(alpha)

    def setStrokeAlpha(self, a):
        if not (isinstance(a, (float, int)) and 0 <= a <= 1):
            raise ValueError('setStrokeAlpha invalid value %r' % a)
        getattr(self, '_setStrokeAlpha', lambda x: None)(a)

    def setFillAlpha(self, a):
        if not (isinstance(a, (float, int)) and 0 <= a <= 1):
            raise ValueError('setFillAlpha invalid value %r' % a)
        getattr(self, '_setFillAlpha', lambda x: None)(a)

    def setStrokeOverprint(self, a):
        getattr(self, '_setStrokeOverprint', lambda x: None)(a)

    def setFillOverprint(self, a):
        getattr(self, '_setFillOverprint', lambda x: None)(a)

    def setOverprintMask(self, a):
        getattr(self, '_setOverprintMask', lambda x: None)(a)