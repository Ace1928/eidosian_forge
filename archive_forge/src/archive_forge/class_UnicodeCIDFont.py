import os
import marshal
import time
from hashlib import md5
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase._cidfontdata import allowedTypeFaces, allowedEncodings, CIDFontInfo, \
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfdoc
from reportlab.lib.rl_accel import escapePDF
from reportlab.rl_config import CMapSearchPath
from reportlab.lib.utils import isSeq, isBytes
class UnicodeCIDFont(CIDFont):
    """Wraps up CIDFont to hide explicit encoding choice;
    encodes text for output as UTF16.

    lang should be one of 'jpn',chs','cht','kor' for now.
    if vertical is set, it will select a different widths array
    and possibly glyphs for some punctuation marks.

    halfWidth is only for Japanese.


    >>> dodgy = UnicodeCIDFont('nonexistent')
    Traceback (most recent call last):
    ...
    KeyError: "don't know anything about CID font nonexistent"
    >>> heisei = UnicodeCIDFont('HeiseiMin-W3')
    >>> heisei.name
    'HeiseiMin-W3'
    >>> heisei.language
    'jpn'
    >>> heisei.encoding.name
    'UniJIS-UCS2-H'
    >>> #This is how PDF data gets encoded.
    >>> print(heisei.formatForPdf('hello'))
    \\000h\\000e\\000l\\000l\\000o
    >>> tokyo = u'東䫬'
    >>> print(heisei.formatForPdf(tokyo))
    gqJ\\354
    >>> print(heisei.stringWidth(tokyo,10))
    20.0
    >>> print(heisei.stringWidth('hello world',10))
    45.83
    """

    def __init__(self, face, isVertical=False, isHalfWidth=False):
        try:
            lang, defaultEncoding = defaultUnicodeEncodings[face]
        except KeyError:
            raise KeyError("don't know anything about CID font %s" % face)
        self.language = lang
        enc = defaultEncoding[:-1]
        if isHalfWidth:
            enc = enc + 'HW-'
        if isVertical:
            enc = enc + 'V'
        else:
            enc = enc + 'H'
        CIDFont.__init__(self, face, enc)
        self.name = self.fontName = face
        self.vertical = isVertical
        self.isHalfWidth = isHalfWidth
        self.unicodeWidths = widthsByUnichar[self.name]

    def formatForPdf(self, text):
        from codecs import utf_16_be_encode
        if isBytes(text):
            text = text.decode('utf8')
        utfText = utf_16_be_encode(text)[0]
        encoded = escapePDF(utfText)
        return encoded

    def stringWidth(self, text, size, encoding=None):
        """Just ensure we do width test on characters, not bytes..."""
        if isBytes(text):
            text = text.decode('utf8')
        widths = self.unicodeWidths
        return size * 0.001 * sum([widths.get(uch, 1000) for uch in text])