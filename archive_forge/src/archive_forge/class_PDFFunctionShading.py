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
class PDFFunctionShading(PDFShading):
    required = PDFShading.required + ('Function',)
    permitted = PDFShading.permitted + ('Domain', 'Matrix', 'Function')

    def __init__(self, Function, ColorSpace, **kw):
        self.Function = Function
        self.ColorSpace = ColorSpace
        self.otherkw = kw

    def Dict(self, document):
        d = {}
        d.update(self.otherkw)
        d['ShadingType'] = 1
        d['ColorSpace'] = PDFName(self.ColorSpace)
        d['Function'] = document.Reference(self.Function)
        return self.ShadingDict(**d)