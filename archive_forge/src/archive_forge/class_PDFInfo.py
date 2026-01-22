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
class PDFInfo(PDFObject):
    """PDF documents can have basic information embedded, viewable from
    File | Document Info in Acrobat Reader.  If this is wrong, you get
    Postscript errors while printing, even though it does not print."""
    producer = _default_producer
    creator = 'ReportLab PDF Library - www.reportlab.com'
    title = 'untitled'
    author = 'anonymous'
    subject = 'unspecified'
    keywords = ''
    _dateFormatter = None

    def __init__(self):
        self.trapped = 'False'

    def digest(self, md5object):
        for x in (self.title, self.author, self.subject, self.keywords):
            md5object.update(bytestr(x))

    def format(self, document):
        D = {}
        D['Title'] = PDFString(self.title)
        D['Author'] = PDFString(self.author)
        D['ModDate'] = D['CreationDate'] = PDFDate(ts=document._timeStamp, dateFormatter=self._dateFormatter)
        D['Producer'] = PDFString(self.producer)
        D['Creator'] = PDFString(self.creator)
        D['Subject'] = PDFString(self.subject)
        D['Keywords'] = PDFString(self.keywords)
        D['Trapped'] = PDFName(self.trapped)
        PD = PDFDictionary(D)
        return PD.format(document)

    def copy(self):
        """shallow copy - useful in pagecatchering"""
        thing = self.__klass__()
        for k, v in self.__dict__.items():
            setattr(thing, k, v)
        return thing