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
class PDFPostScriptXObject(PDFObject):
    """For embedding PD (e.g. tray commands) in PDF"""

    def __init__(self, content=None):
        self.content = content

    def format(self, document):
        S = PDFStream()
        S.content = self.content
        S.__Comment__ = 'xobject postscript stream'
        sdict = S.dictionary
        sdict['Type'] = PDFName('XObject')
        sdict['Subtype'] = PDFName('PS')
        return S.format(document)