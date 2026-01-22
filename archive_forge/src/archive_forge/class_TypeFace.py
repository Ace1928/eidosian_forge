import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
class TypeFace:

    def __init__(self, name):
        self.name = name
        self.glyphNames = []
        self.glyphWidths = {}
        self.ascent = 0
        self.descent = 0
        self.familyName = None
        self.bold = 0
        self.italic = 0
        if name == 'ZapfDingbats':
            self.requiredEncoding = 'ZapfDingbatsEncoding'
        elif name == 'Symbol':
            self.requiredEncoding = 'SymbolEncoding'
        else:
            self.requiredEncoding = None
        if name in standardFonts:
            self.builtIn = 1
            self._loadBuiltInData(name)
        else:
            self.builtIn = 0

    def _loadBuiltInData(self, name):
        """Called for the built in 14 fonts.  Gets their glyph data.
        We presume they never change so this can be a shared reference."""
        name = str(name)
        self.glyphWidths = _fontdata.widthsByFontGlyph[name]
        self.glyphNames = list(self.glyphWidths.keys())
        self.ascent, self.descent = _fontdata.ascent_descent[name]

    def getFontFiles(self):
        """Info function, return list of the font files this depends on."""
        return []

    def findT1File(self, ext='.pfb'):
        possible_exts = (ext.lower(), ext.upper())
        if hasattr(self, 'pfbFileName'):
            r_basename = os.path.splitext(self.pfbFileName)[0]
            for e in possible_exts:
                if rl_isfile(r_basename + e):
                    return r_basename + e
        try:
            r = _fontdata.findT1File(self.name)
        except:
            afm = bruteForceSearchForAFM(self.name)
            if afm:
                if ext.lower() == '.pfb':
                    for e in possible_exts:
                        pfb = os.path.splitext(afm)[0] + e
                        if rl_isfile(pfb):
                            r = pfb
                        else:
                            r = None
                elif ext.lower() == '.afm':
                    r = afm
            else:
                r = None
        if r is None:
            warnOnce("Can't find %s for face '%s'" % (ext, self.name))
        return r