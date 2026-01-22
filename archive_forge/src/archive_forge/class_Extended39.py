from reportlab.lib.units import inch
from reportlab.lib.utils import asNative
from reportlab.graphics.barcode.common import Barcode
from string import ascii_uppercase, ascii_lowercase, digits as string_digits
class Extended39(_Code39Base):
    """
    Extended Code 39 is a convention for encoding additional characters
    not present in stanmdard Code 39 by using pairs of characters to
    represent the characters missing in Standard Code 39.

    See Standard39 for arguments.

    Sources of Information on Extended Code 39:

    http://www.semiconductor.agilent.com/barcode/sg/Misc/xcode_39.html
    http://www.barcodeman.com/c39_ext.html
    """

    def validate(self):
        vval = ''
        self.valid = 1
        for c in self.value:
            if c not in _extchrs:
                self.valid = 0
                continue
            vval = vval + c
        self.validated = vval
        return vval

    def encode(self):
        self.encoded = ''
        for c in self.validated:
            if c in _extended:
                self.encoded = self.encoded + _extended[c]
            elif c in _stdchrs:
                self.encoded = self.encoded + c
            else:
                raise ValueError
        self.encoded = _encode39(self.encoded, self.checksum, self.stop)
        return self.encoded