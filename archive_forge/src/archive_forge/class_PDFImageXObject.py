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
class PDFImageXObject(PDFObject):

    def __init__(self, name, source=None, mask=None):
        self.name = name
        self.width = 24
        self.height = 23
        self.bitsPerComponent = 1
        self.colorSpace = 'DeviceGray'
        self._filters = rl_config.useA85 and ('ASCII85Decode',) or ()
        self.streamContent = '\n            003B00 002700 002480 0E4940 114920 14B220 3CB650\n            75FE88 17FF8C 175F14 1C07E2 3803C4 703182 F8EDFC\n            B2BBC2 BB6F84 31BFC2 18EA3C 0E3E00 07FC00 03F800\n            1E1800 1FF800>\n            '
        self.mask = mask
        if source is None:
            pass
        elif hasattr(source, 'jpeg_fh'):
            self.loadImageFromSRC(source)
        else:
            import os
            ext = os.path.splitext(source)[1].lower()
            src = open_for_read(source)
            try:
                if not (ext in ('.jpg', '.jpeg') and self.loadImageFromJPEG(src)):
                    if rl_config.useA85:
                        self.loadImageFromA85(src)
                    else:
                        self.loadImageFromRaw(src)
            finally:
                src.close()

    def loadImageFromA85(self, source):
        IMG = []
        imagedata = pdfutils.makeA85Image(source, IMG=IMG, detectJpeg=True)
        if not imagedata:
            return self.loadImageFromSRC(IMG[0])
        imagedata = [s.strip() for s in imagedata]
        words = imagedata[1].split()
        self.width, self.height = (int(words[1]), int(words[3]))
        self.colorSpace = {'/RGB': 'DeviceRGB', '/G': 'DeviceGray', '/CMYK': 'DeviceCMYK'}[words[7]]
        self.bitsPerComponent = 8
        self._filters = ('ASCII85Decode', 'FlateDecode')
        if IMG:
            self._checkTransparency(IMG[0])
        elif self.mask == 'auto':
            self.mask = None
        self.streamContent = ''.join(imagedata[3:-1])

    def loadImageFromJPEG(self, imageFile):
        try:
            try:
                info = pdfutils.readJPEGInfo(imageFile)
            finally:
                imageFile.seek(0)
        except:
            return False
        self.width, self.height = (info[0], info[1])
        self.bitsPerComponent = 8
        if info[2] == 1:
            self.colorSpace = 'DeviceGray'
        elif info[2] == 3:
            self.colorSpace = 'DeviceRGB'
        else:
            self.colorSpace = 'DeviceCMYK'
            self._dotrans = 1
        self.streamContent = imageFile.read()
        if rl_config.useA85:
            self.streamContent = asciiBase85Encode(self.streamContent)
            self._filters = ('ASCII85Decode', 'DCTDecode')
        else:
            self._filters = ('DCTDecode',)
        self.mask = None
        return True

    def loadImageFromRaw(self, source):
        IMG = []
        imagedata = pdfutils.makeRawImage(source, IMG=IMG, detectJpeg=True)
        if not imagedata:
            return self.loadImageFromSRC(IMG[0])
        words = imagedata[1].split()
        self.width = int(words[1])
        self.height = int(words[3])
        self.colorSpace = {'/RGB': 'DeviceRGB', '/G': 'DeviceGray', '/CMYK': 'DeviceCMYK'}[words[7]]
        self.bitsPerComponent = 8
        self._filters = ('FlateDecode',)
        if IMG:
            self._checkTransparency(IMG[0])
        elif self.mask == 'auto':
            self.mask = None
        self.streamContent = b''.join(imagedata[3:-1])

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

    def loadImageFromSRC(self, im):
        """Extracts the stream, width and height"""
        fp = im.jpeg_fh()
        if fp:
            self.loadImageFromJPEG(fp)
        else:
            self.width, self.height = im.getSize()
            raw = im.getRGBData()
            self.streamContent = zlib.compress(raw)
            if rl_config.useA85:
                self.streamContent = asciiBase85Encode(self.streamContent)
                self._filters = ('ASCII85Decode', 'FlateDecode')
            else:
                self._filters = ('FlateDecode',)
            self.colorSpace = _mode2CS[im.mode]
            self.bitsPerComponent = 8
            self._checkTransparency(im)

    def format(self, document):
        S = PDFStream(content=self.streamContent)
        dict = S.dictionary
        dict['Type'] = PDFName('XObject')
        dict['Subtype'] = PDFName('Image')
        dict['Width'] = self.width
        dict['Height'] = self.height
        dict['BitsPerComponent'] = self.bitsPerComponent
        dict['ColorSpace'] = PDFName(self.colorSpace)
        if self.colorSpace == 'DeviceCMYK' and getattr(self, '_dotrans', 0):
            dict['Decode'] = PDFArray([1, 0, 1, 0, 1, 0, 1, 0])
        elif getattr(self, '_decode', None):
            dict['Decode'] = PDFArray(self._decode)
        dict['Filter'] = PDFArray(map(PDFName, self._filters))
        dict['Length'] = len(self.streamContent)
        if self.mask:
            dict['Mask'] = PDFArray(self.mask)
        if getattr(self, 'smask', None):
            dict['SMask'] = self.smask
        return S.format(document)