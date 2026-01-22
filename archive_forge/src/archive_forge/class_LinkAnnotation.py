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
class LinkAnnotation(Annotation):
    permitted = Annotation.permitted + ('Dest', 'A', 'PA')

    def __init__(self, Rect, Contents, Destination, Border='[0 0 1]', **kw):
        self.Border = Border
        self.Rect = Rect
        self.Contents = Contents
        self.Destination = Destination
        self.otherkw = kw

    def dummyDictString(self):
        return '\n          << /Type /Annot /Subtype /Link /Rect [71 717 190 734] /Border [16 16 1]\n             /Dest [23 0 R /Fit] >>\n             '

    def Dict(self):
        d = {}
        d.update(self.otherkw)
        d['Border'] = self.Border
        d['Rect'] = self.Rect
        d['Contents'] = self.Contents
        d['Subtype'] = '/Link'
        d['Dest'] = self.Destination
        return self.AnnotationDict(**d)