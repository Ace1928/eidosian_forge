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
def testpage(document):
    P = PDFPage()
    P.Contents = teststream()
    pages = document.Pages
    P.Parent = document.Reference(pages)
    P.MediaBox = PDFArray([0, 0, 595, 841])
    resources = PDFResourceDictionary()
    resources.allProcs()
    resources.basicFonts()
    P.Resources = resources
    pages.addPage(P)