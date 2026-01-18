from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import open_and_read, isBytes, rl_exec
from .shapes import _baseGFontName, _PATH_OP_ARG_COUNT, _PATH_OP_NAMES, definePath
from sys import exc_info
def text2Path(text, x=0, y=0, fontName=_baseGFontName, fontSize=1000, anchor='start', truncate=1, pathReverse=0, gs=None, **kwds):
    t2pd = kwds.pop('text2PathDescription', text2PathDescription)
    return definePath(t2pd(text, x=x, y=y, fontName=fontName, fontSize=fontSize, anchor=anchor, truncate=truncate, pathReverse=pathReverse, gs=gs), **kwds)