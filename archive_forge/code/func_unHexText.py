import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
def unHexText(hexText):
    equalityCheck(hexText[0], '<', 'bad hex text')
    equalityCheck(hexText[-1], '>', 'bad hex text')
    return unhexlify(hexText[1:-1])