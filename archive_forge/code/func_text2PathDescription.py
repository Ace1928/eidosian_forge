from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import open_and_read, isBytes, rl_exec
from .shapes import _baseGFontName, _PATH_OP_ARG_COUNT, _PATH_OP_NAMES, definePath
from sys import exc_info
def text2PathDescription(text, x=0, y=0, fontName=_baseGFontName, fontSize=1000, anchor='start', truncate=1, pathReverse=0, gs=None):
    """_renderPM text2PathDescription(text, x=0, y=0, fontName='fontname',
                                    fontSize=1000, font = 'fontName',
                                    anchor='start', truncate=1, pathReverse=0, gs=None)
                """
    font = getFont(fontName)
    if font._multiByte and (not font._dynamicFont):
        raise ValueError("text2PathDescription doesn't support multi byte fonts like %r" % fontName)
    P_extend = [].extend
    if not anchor == 'start':
        textLen = stringWidth(text, fontName, fontSize)
        if anchor == 'end':
            x = x - textLen
        elif anchor == 'middle':
            x = x - textLen / 2.0
    if gs is None:
        from _rl_renderPM import gstate
        gs = gstate(1, 1)
    setFont(gs, fontName, fontSize)
    if font._dynamicFont:
        for g in gs._stringPath(text, x, y):
            P_extend(processGlyph(g, truncate=truncate, pathReverse=pathReverse))
    else:
        if isBytes(text):
            try:
                text = text.decode('utf8')
            except UnicodeDecodeError as e:
                i, j = e.args[2:4]
                raise UnicodeDecodeError(*e.args[:4] + ('%s\n%s-->%s<--%s' % (e.args[4], text[max(i - 10, 0):i], text[i:j], text[j:j + 10]),))
        fc = font
        FT = unicode2T1(text, [font] + font.substitutionFonts)
        nm1 = len(FT) - 1
        for i, (f, t) in enumerate(FT):
            if f != fc:
                setFont(gs, f.fontName, fontSize)
                fc = f
            for g in gs._stringPath(t, x, y):
                P_extend(processGlyph(g, truncate=truncate, pathReverse=pathReverse))
            if i != nm1:
                x += f.stringWidth(t.decode(f.encName), fontSize)
    return P_extend.__self__