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
def _checkTransparency(self, im):
    if self.mask == 'auto':
        if im._dataA:
            self.mask = None
            self._smask = PDFImageXObject(_digester(im._dataA.getRGBData()), im._dataA, mask=None)
            self._smask._decode = [0, 1]
        else:
            tc = im.getTransparent()
            if tc:
                self.mask = (tc[0], tc[0], tc[1], tc[1], tc[2], tc[2])
            else:
                self.mask = None
    elif hasattr(self.mask, 'rgb'):
        _ = self.mask.rgb()
        self.mask = (_[0], _[0], _[1], _[1], _[2], _[2])