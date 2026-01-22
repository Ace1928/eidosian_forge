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
class CheckedPDFDictionary(PDFDictionary):
    validate = {}

    def __init__(self, dict=None, validate=None):
        PDFDictionary.__init__(self, dict)
        if validate:
            self.validate = validate

    def __setitem__(self, name, value):
        if name not in self.validate:
            raise ValueError('invalid key, %r' % name)
        cvalue = self.validate[name](value)
        if cvalue is None:
            raise ValueError('Bad value %r for key %r' % (value, name))
        PDFDictionary.__setitem__(self, name, cvalue)