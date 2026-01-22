from fontTools.misc.textTools import bytesjoin, strjoin, tobytes, tostr, safeEval
from fontTools.misc import sstruct
from . import DefaultTable
import base64
class SignatureRecord(object):

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.__dict__)

    def toXML(self, writer, ttFont):
        writer.begintag(self.__class__.__name__, format=self.ulFormat)
        writer.newline()
        writer.write_noindent('-----BEGIN PKCS7-----\n')
        writer.write_noindent(b64encode(self.pkcs7))
        writer.write_noindent('-----END PKCS7-----\n')
        writer.endtag(self.__class__.__name__)

    def fromXML(self, name, attrs, content, ttFont):
        self.ulFormat = safeEval(attrs['format'])
        self.usReserved1 = safeEval(attrs.get('reserved1', '0'))
        self.usReserved2 = safeEval(attrs.get('reserved2', '0'))
        self.pkcs7 = base64.b64decode(tobytes(strjoin(filter(pem_spam, content))))