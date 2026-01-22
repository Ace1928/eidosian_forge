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
class PDFCrossReferenceSubsection(PDFObject):

    def __init__(self, firstentrynumber, idsequence):
        self.firstentrynumber = firstentrynumber
        self.idsequence = idsequence

    def format(self, document):
        """id sequence should represent contiguous object nums else error. free numbers not supported (yet)"""
        firstentrynumber = self.firstentrynumber
        idsequence = self.idsequence
        entries = list(idsequence)
        nentries = len(idsequence)
        taken = {}
        if firstentrynumber == 0:
            taken[0] = 'standard free entry'
            nentries = nentries + 1
            entries.insert(0, '0000000000 65535 f ')
        idToNV = document.idToObjectNumberAndVersion
        idToOffset = document.idToOffset
        lastentrynumber = firstentrynumber + nentries - 1
        for id in idsequence:
            num, version = idToNV[id]
            if num in taken:
                raise ValueError('object number collision %s %s %s' % (num, repr(id), repr(taken[id])))
            if num > lastentrynumber or num < firstentrynumber:
                raise ValueError('object number %s not in range %s..%s' % (num, firstentrynumber, lastentrynumber))
            rnum = num - firstentrynumber
            taken[num] = id
            offset = idToOffset[id]
            entries[num] = '%0.10d %0.5d n ' % (offset, version)
        firstline = '%s %s' % (firstentrynumber, nentries)
        entries.insert(0, firstline)
        entries.append('')
        return pdfdocEnc('\n'.join(entries))