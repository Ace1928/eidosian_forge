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
def teststream(content=None):
    if content is None:
        content = teststreamcontent
    content = content.strip() + '\n'
    S = PDFStream(content=content, filters=rl_config.useA85 and [PDFBase85Encode, PDFZCompress] or [PDFZCompress])
    S.__Comment__ = 'test stream'
    return S