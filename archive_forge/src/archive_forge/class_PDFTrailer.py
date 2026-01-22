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
class PDFTrailer(PDFObject):

    def __init__(self, startxref, Size=None, Prev=None, Root=None, Info=None, ID=None, Encrypt=None):
        self.startxref = startxref
        if Size is None or Root is None:
            raise ValueError('Size and Root keys required')
        dict = self.dict = PDFDictionary()
        for n, v in [('Size', Size), ('Prev', Prev), ('Root', Root), ('Info', Info), ('ID', ID), ('Encrypt', Encrypt)]:
            if v is not None:
                dict[n] = v
        dict.multiline = 'forced'

    def format(self, document):
        fdict = self.dict.format(document, IND=b'\n')
        return b''.join([b'trailer\n', fdict, b'\nstartxref\n', pdfdocEnc(str(self.startxref)), b'\n%%EOF\n'])