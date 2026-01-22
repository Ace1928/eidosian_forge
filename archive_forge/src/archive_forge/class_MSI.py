from reportlab.platypus.flowables import Flowable
from reportlab.lib.units import inch
from string import ascii_lowercase, ascii_uppercase, digits as string_digits
class MSI(Barcode):
    """
    MSI is a numeric-only barcode.

    Options that may be passed to constructor:

        value (int, or numeric string required.):
            The value to encode.

        barWidth (float, default .0075):
            X-Dimension, or width of the smallest element

        ratio (float, default 2.2):
            The ratio of wide elements to narrow elements.

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

        lquiet (float, see default below):
            Quiet zone size to left of code, if quiet is true.
            Default is the greater of .25 inch, or 10 barWidths.

        rquiet (float, defaults as above):
            Quiet zone size to right left of code, if quiet is true.

        stop (bool, default 1):
            Whether to include start/stop symbols.

    Sources of Information on MSI Bar Code:

    http://www.semiconductor.agilent.com/barcode/sg/Misc/msi_code.html
    http://www.adams1.com/pub/russadam/plessy.html
    """
    patterns = {'start': 'Bs', 'stop': 'bSb', '0': 'bSbSbSbS', '1': 'bSbSbSBs', '2': 'bSbSBsbS', '3': 'bSbSBsBs', '4': 'bSBsbSbS', '5': 'bSBsbSBs', '6': 'bSBsBsbS', '7': 'bSBsBsBs', '8': 'BsbSbSbS', '9': 'BsbSbSBs'}
    stop = 1
    barHeight = None
    barWidth = inch * 0.0075
    ratio = 2.2
    checksum = 1
    bearers = 0.0
    quiet = 1
    lquiet = None
    rquiet = None

    def __init__(self, value='', **args):
        if type(value) == type(1):
            value = str(value)
        for k, v in args.items():
            setattr(self, k, v)
        if self.quiet:
            if self.lquiet is None:
                self.lquiet = max(inch * 0.25, self.barWidth * 10.0)
                self.rquiet = max(inch * 0.25, self.barWidth * 10.0)
        else:
            self.lquiet = self.rquiet = 0.0
        Barcode.__init__(self, value)

    def validate(self):
        vval = ''
        self.valid = 1
        for c in self.value.strip():
            if c not in string_digits:
                self.valid = 0
                continue
            vval = vval + c
        self.validated = vval
        return vval

    def encode(self):
        s = self.validated
        if self.checksum:
            c = ''
            for i in range(1, len(s), 2):
                c = c + s[i]
            d = str(int(c) * 2)
            t = 0
            for c in d:
                t = t + int(c)
            for i in range(0, len(s), 2):
                t = t + int(s[i])
            c = 10 - t % 10
            s = s + str(c)
        self.encoded = s

    def decompose(self):
        dval = self.stop and [self.patterns['start']] or []
        dval += [self.patterns[c] for c in self.encoded]
        if self.stop:
            dval.append(self.patterns['stop'])
        self.decomposed = ''.join(dval)
        return self.decomposed