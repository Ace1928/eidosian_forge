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
class PDFFormXObject(PDFObject):
    XObjects = Annots = BBox = Matrix = Contents = stream = Resources = None
    hasImages = 1
    compression = 0

    def __init__(self, lowerx, lowery, upperx, uppery):
        self.lowerx = lowerx
        self.lowery = lowery
        self.upperx = upperx
        self.uppery = uppery

    def setStreamList(self, data):
        if isSeq(data):
            data = '\n'.join(data)
        self.stream = pdfdocEnc(data)

    def BBoxList(self):
        """get the declared bounding box for the form as a list"""
        if self.BBox:
            return list(self.BBox.sequence)
        else:
            return [self.lowerx, self.lowery, self.upperx, self.uppery]

    def format(self, document):
        self.BBox = self.BBox or PDFArray([self.lowerx, self.lowery, self.upperx, self.uppery])
        self.Matrix = self.Matrix or PDFArray([1, 0, 0, 1, 0, 0])
        if not self.Annots:
            self.Annots = None
        else:
            raise ValueError("annotations don't work in PDFFormXObjects yet")
        if not self.Contents:
            stream = self.stream
            if not stream:
                self.Contents = teststream()
            else:
                S = PDFStream()
                S.content = stream
                S.__Comment__ = 'xobject form stream'
                self.Contents = S
        if not self.Resources:
            resources = PDFResourceDictionary()
            resources.basicFonts()
            if self.hasImages:
                resources.allProcs()
            else:
                resources.basicProcs()
            if self.XObjects:
                resources.XObject = self.XObjects
            self.Resources = resources
        if self.compression:
            self.Contents.filters = rl_config.useA85 and [PDFBase85Encode, PDFZCompress] or [PDFZCompress]
        sdict = self.Contents.dictionary
        sdict['Type'] = PDFName('XObject')
        sdict['Subtype'] = PDFName('Form')
        sdict['FormType'] = 1
        sdict['BBox'] = self.BBox
        sdict['Matrix'] = self.Matrix
        sdict['Resources'] = self.Resources
        return self.Contents.format(document)