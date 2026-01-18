import sys, os
from binascii import hexlify, unhexlify
from hashlib import md5
from io import BytesIO
from reportlab.lib.utils import asBytes, int2Byte, rawBytes, asNative
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfdoc import PDFObject
from reportlab.platypus.flowables import Flowable
from reportlab import rl_config
def xorKey(num, key):
    """xor's each byte of the key with the number, which is <256"""
    if num == 0:
        return key
    return bytes((num ^ k for k in key))