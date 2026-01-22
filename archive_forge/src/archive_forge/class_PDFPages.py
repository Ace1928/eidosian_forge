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
class PDFPages(PDFCatalog):
    """PAGES TREE WITH ONE INTERNAL NODE, FOR "BALANCING" CHANGE IMPLEMENTATION"""
    __Comment__ = 'page tree'
    __RefOnly__ = 1
    __Defaults__ = {'Type': PDFName('Pages')}
    __NoDefault__ = 'Kids Count Parent'.split()
    __Refs__ = ['Parent']

    def __init__(self):
        self.pages = []

    def __getitem__(self, item):
        return self.pages[item]

    def addPage(self, page):
        self.pages.append(page)

    def check_format(self, document):
        pages = self.pages
        kids = PDFArray(pages)
        kids.References(document)
        self.Kids = kids
        self.Count = len(pages)