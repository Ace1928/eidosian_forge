from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def _formatText(self, text):
    """Generates PDF text output operator(s)"""
    if log2vis and self.direction in ('LTR', 'RTL'):
        text = log2vis(text, directionsMap.get(self.direction, DIR_ON), clean=True)
    canv = self._canvas
    font = pdfmetrics.getFont(self._fontname)
    R = []
    if font._dynamicFont:
        for subset, t in font.splitString(text, canv._doc):
            if subset != self._curSubset:
                pdffontname = font.getSubsetInternalName(subset, canv._doc)
                R.append('%s %s Tf %s TL' % (pdffontname, fp_str(self._fontsize), fp_str(self._leading)))
                self._curSubset = subset
            R.append('(%s) Tj' % canv._escape(t))
    elif font._multiByte:
        R.append('%s %s Tf %s TL' % (canv._doc.getInternalFontName(font.fontName), fp_str(self._fontsize), fp_str(self._leading)))
        R.append('(%s) Tj' % font.formatForPdf(text))
    else:
        fc = font
        if isBytes(text):
            try:
                text = text.decode('utf8')
            except UnicodeDecodeError as e:
                i, j = e.args[2:4]
                raise UnicodeDecodeError(*e.args[:4] + ('%s\n%s-->%s<--%s' % (e.args[4], text[max(i - 10, 0):i], text[i:j], text[j:j + 10]),))
        for f, t in pdfmetrics.unicode2T1(text, [font] + font.substitutionFonts):
            if f != fc:
                R.append('%s %s Tf %s TL' % (canv._doc.getInternalFontName(f.fontName), fp_str(self._fontsize), fp_str(self._leading)))
                fc = f
            R.append('(%s) Tj' % canv._escape(t))
        if font != fc:
            R.append('%s %s Tf %s TL' % (canv._doc.getInternalFontName(self._fontname), fp_str(self._fontsize), fp_str(self._leading)))
    return ' '.join(R)