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
class PDFStreamFilterBase85Encode:
    pdfname = 'ASCII85Decode'

    def encode(self, text):
        from reportlab.pdfbase.pdfutils import _wrap
        text = asciiBase85Encode(text)
        if rl_config.wrapA85:
            text = _wrap(text)
        return text

    def decode(self, text):
        return asciiBase85Decode(text)