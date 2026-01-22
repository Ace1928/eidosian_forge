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
class PDFStitchingFunction(PDFFunction):
    required = PDFFunction.required + ('Functions', 'Bounds', 'Encode')
    permitted = PDFFunction.permitted + ('Functions', 'Bounds', 'Encode')

    def __init__(self, Functions, Bounds, Encode, **kw):
        self.Functions = Functions
        self.Bounds = Bounds
        self.Encode = Encode
        self.otherkw = kw

    def Dict(self, document):
        d = {}
        d.update(self.otherkw)
        d['FunctionType'] = 3
        d['Functions'] = PDFArray([document.Reference(x) for x in self.Functions])
        d['Bounds'] = PDFArray(self.Bounds)
        d['Encode'] = PDFArray(self.Encode)
        return self.FunctionDict(**d)