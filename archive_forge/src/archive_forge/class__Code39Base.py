from reportlab.lib.units import inch
from reportlab.lib.utils import asNative
from reportlab.graphics.barcode.common import Barcode
from string import ascii_uppercase, ascii_lowercase, digits as string_digits
class _Code39Base(Barcode):
    barWidth = inch * 0.0075
    lquiet = None
    rquiet = None
    quiet = 1
    gap = None
    barHeight = None
    ratio = 2.2
    checksum = 1
    bearers = 0.0
    stop = 1

    def __init__(self, value='', **args):
        value = asNative(value)
        for k, v in args.items():
            setattr(self, k, v)
        if self.quiet:
            if self.lquiet is None:
                self.lquiet = max(inch * 0.25, self.barWidth * 10.0)
                self.rquiet = max(inch * 0.25, self.barWidth * 10.0)
        else:
            self.lquiet = self.rquiet = 0.0
        Barcode.__init__(self, value)

    def decompose(self):
        dval = ''
        for c in self.encoded:
            dval = dval + _patterns[c][0] + 'i'
        self.decomposed = dval[:-1]
        return self.decomposed

    def _humanText(self):
        return self.stop and self.encoded[1:-1] or self.encoded