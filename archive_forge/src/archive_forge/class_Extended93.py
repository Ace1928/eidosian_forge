from reportlab.lib.units import inch
from reportlab.lib.utils import asNative
from reportlab.graphics.barcode.common import MultiWidthBarcode
class Extended93(_Code93Base):
    """
    Extended Code 93 is a convention for encoding the entire 128 character
    set using pairs of characters to represent the characters missing in
    Standard Code 93. It is very much like Extended Code 39 in that way.
    
    See Standard93 for arguments.
    """

    def validate(self):
        vval = []
        self.valid = 1
        a = vval.append
        for c in self.value:
            if c not in _patterns and c not in _extended:
                self.valid = 0
                continue
            a(c)
        self.validated = ''.join(vval)
        return self.validated

    def encode(self):
        self.encoded = ''
        for c in self.validated:
            if c in _patterns:
                self.encoded = self.encoded + c
            elif c in _extended:
                self.encoded = self.encoded + _extended[c]
            else:
                raise ValueError
        self.encoded = _encode93(self.encoded)
        return self.encoded

    def _humanText(self):
        return self.validated + self.encoded[-2:]