from reportlab.lib.units import inch
from reportlab.lib.utils import asNative
from reportlab.graphics.barcode.common import Barcode
from string import ascii_uppercase, ascii_lowercase, digits as string_digits
class Standard39(_Code39Base):
    """
    Options that may be passed to constructor:

        value (int, or numeric string required.):
            The value to encode.

        barWidth (float, default .0075):
            X-Dimension, or width of the smallest element
            Minumum is .0075 inch (7.5 mils).

        ratio (float, default 2.2):
            The ratio of wide elements to narrow elements.
            Must be between 2.0 and 3.0 (or 2.2 and 3.0 if the
            barWidth is greater than 20 mils (.02 inch))

        gap (float or None, default None):
            width of intercharacter gap. None means "use barWidth".

        barHeight (float, see default below):
            Height of the symbol.  Default is the height of the two
            bearer bars (if they exist) plus the greater of .25 inch
            or .15 times the symbol's length.

        checksum (bool, default 1):
            Wether to compute and include the check digit

        bearers (float, in units of barWidth. default 0):
            Height of bearer bars (horizontal bars along the top and
            bottom of the barcode). Default is 0 (no bearers).

        quiet (bool, default 1):
            Wether to include quiet zones in the symbol.

        lquiet (float, see default below):
            Quiet zone size to left of code, if quiet is true.
            Default is the greater of .25 inch, or .15 times the symbol's
            length.

        rquiet (float, defaults as above):
            Quiet zone size to right left of code, if quiet is true.

        stop (bool, default 1):
            Whether to include start/stop symbols.

    Sources of Information on Code 39:

    http://www.semiconductor.agilent.com/barcode/sg/Misc/code_39.html
    http://www.adams1.com/pub/russadam/39code.html
    http://www.barcodeman.com/c39_1.html

    Official Spec, "ANSI/AIM BC1-1995, USS" is available for US$45 from
    http://www.aimglobal.org/aimstore/
    """

    def validate(self):
        vval = [].append
        self.valid = 1
        for c in self.value:
            if c in ascii_lowercase:
                c = c.upper()
            if c not in _stdchrs:
                self.valid = 0
                continue
            vval(c)
        self.validated = ''.join(vval.__self__)
        return self.validated

    def encode(self):
        self.encoded = _encode39(self.validated, self.checksum, self.stop)
        return self.encoded