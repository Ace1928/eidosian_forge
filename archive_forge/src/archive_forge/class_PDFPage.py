import binascii, codecs, zlib
from collections import OrderedDict
from reportlab.pdfbase import pdfutils
from reportlab import rl_config
from reportlab.lib.utils import open_for_read, makeFileName, isSeq, isBytes, isUnicode, _digester, isStr, bytestr, annotateException, TimeStamp
from reportlab.lib.rl_accel import escapePDF, fp_str, asciiBase85Encode, asciiBase85Decode
from reportlab.pdfbase import pdfmetrics
from hashlib import md5
from sys import stderr
import re
class PDFPage(PDFCatalog):
    __Comment__ = 'Page dictionary'
    Override_default_compilation = 0
    __RefOnly__ = 1
    __Defaults__ = {'Type': PDFName('Page')}
    __NoDefault__ = 'Parent\n        MediaBox Resources Contents CropBox Rotate Thumb Annots B Dur Hid Trans AA\n        PieceInfo LastModified SeparationInfo ArtBox TrimBox BleedBox ID PZ\n        Trans'.split()
    __Refs__ = 'Contents Parent ID'.split()
    pagewidth = 595
    pageheight = 842
    stream = None
    hasImages = 0
    compression = 0
    XObjects = None
    _colorsUsed = {}
    _shadingsUsed = {}
    Trans = None

    def __init__(self):
        for name in self.__NoDefault__:
            setattr(self, name, None)

    def setCompression(self, onoff):
        self.compression = onoff

    def setStream(self, code):
        if self.Override_default_compilation:
            raise ValueError('overridden! must set stream explicitly')
        if isSeq(code):
            code = '\n'.join(code) + '\n'
        self.stream = code

    def setPageTransition(self, tranDict):
        self.Trans = PDFDictionary(tranDict)

    def check_format(self, document):
        if self.Override_default_compilation:
            return
        self.MediaBox = self.MediaBox or PDFArray(self.Rotate in (90, 270) and [0, 0, self.pageheight, self.pagewidth] or [0, 0, self.pagewidth, self.pageheight])
        if not self.Annots:
            self.Annots = None
        elif not isinstance(self.Annots, PDFObject):
            self.Annots = PDFArray(self.Annots)
        if not self.Contents:
            stream = self.stream
            if not stream:
                self.Contents = teststream()
            else:
                S = PDFStream()
                if self.compression:
                    S.filters = rl_config.useA85 and [PDFBase85Encode, PDFZCompress] or [PDFZCompress]
                S.content = stream
                S.__Comment__ = 'page stream'
                self.Contents = S
        if not self.Resources:
            resources = PDFResourceDictionary()
            resources.basicFonts()
            if self.hasImages:
                resources.allProcs()
            else:
                resources.basicProcs()
            if self.XObjects:
                resources.XObject = self.XObjects
            if self.ExtGState:
                resources.ExtGState = self.ExtGState
            resources.setShading(self._shadingUsed)
            resources.setColorSpace(self._colorsUsed)
            self.Resources = resources
        if not self.Parent:
            pages = document.Pages
            self.Parent = document.Reference(pages)