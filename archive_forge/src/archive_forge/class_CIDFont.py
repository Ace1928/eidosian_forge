import os
import marshal
import time
from hashlib import md5
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase._cidfontdata import allowedTypeFaces, allowedEncodings, CIDFontInfo, \
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfdoc
from reportlab.lib.rl_accel import escapePDF
from reportlab.rl_config import CMapSearchPath
from reportlab.lib.utils import isSeq, isBytes
class CIDFont(pdfmetrics.Font):
    """Represents a built-in multi-byte font"""
    _multiByte = 1

    def __init__(self, face, encoding):
        assert face in allowedTypeFaces, "TypeFace '%s' not supported! Use any of these instead: %s" % (face, allowedTypeFaces)
        self.faceName = face
        self.face = CIDTypeFace(face)
        assert encoding in allowedEncodings, "Encoding '%s' not supported!  Use any of these instead: %s" % (encoding, allowedEncodings)
        self.encodingName = encoding
        self.encoding = CIDEncoding(encoding)
        self.fontName = self.faceName + '-' + self.encodingName
        self.name = self.fontName
        self.isVertical = self.encodingName[-1] == 'V'
        self.substitutionFonts = []

    def formatForPdf(self, text):
        encoded = escapePDF(text)
        return encoded

    def stringWidth(self, text, size, encoding=None):
        """This presumes non-Unicode input.  UnicodeCIDFont wraps it for that context"""
        cidlist = self.encoding.translate(text)
        if self.isVertical:
            return len(cidlist) * size
        else:
            w = 0
            for cid in cidlist:
                w = w + self.face.getCharWidth(cid)
            return 0.001 * w * size

    def addObjects(self, doc):
        """The explicit code in addMinchoObjects and addGothicObjects
        will be replaced by something that pulls the data from
        _cidfontdata.py in the next few days."""
        internalName = 'F' + repr(len(doc.fontMapping) + 1)
        bigDict = CIDFontInfo[self.face.name]
        bigDict['Name'] = '/' + internalName
        bigDict['Encoding'] = '/' + self.encodingName
        cidObj = structToPDF(bigDict)
        r = doc.Reference(cidObj, internalName)
        fontDict = doc.idToObject['BasicFonts'].dict
        fontDict[internalName] = r
        doc.fontMapping[self.name] = '/' + internalName