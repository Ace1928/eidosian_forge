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
def SaveToFile(self, filename, canvas):
    if getattr(self, '_savedToFile', False):
        raise RuntimeError('class %s instances can only be saved once' % self.__class__.__name__)
    self._savedToFile = True
    if hasattr(getattr(filename, 'write', None), '__call__'):
        myfile = 0
        f = filename
        filename = getattr(f, 'name', None)
        if isinstance(filename, int):
            filename = '<os fd:%d>' % filename
        elif not isStr(filename):
            filename = '<%s@0X%8.8X>' % (f.__class__.__name__, id(f))
        filename = makeFileName(filename)
    elif isStr(filename):
        myfile = 1
        filename = makeFileName(filename)
        f = open(filename, 'wb')
    else:
        raise TypeError('Cannot use %s as a filename or file' % repr(filename))
    data = self.GetPDFData(canvas)
    if isUnicode(data):
        data = data.encode('latin1')
    f.write(data)
    if myfile:
        f.close()
        import os
        if os.name == 'mac':
            from reportlab.lib.utils import markfilename
            markfilename(filename)
    if getattr(canvas, '_verbosity', None):
        print('saved %s' % (filename,))