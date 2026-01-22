from reportlab.platypus.flowables import Flowable
from reportlab.lib.units import inch
from string import ascii_lowercase, ascii_uppercase, digits as string_digits
class I2of5(Barcode):
    """
    Interleaved 2 of 5 is a numeric-only barcode.  It encodes an even
    number of digits; if an odd number is given, a 0 is prepended.

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
            Whether to compute and include the check digit

        bearers (float, in units of barWidth. default 3.0):
            Height of bearer bars (horizontal bars along the top and
            bottom of the barcode). Default is 3 x-dimensions.
            Set to zero for no bearer bars. (Bearer bars help detect
            misscans, so it is suggested to leave them on).

        bearerBox (bool default False)
            if true draw a  true rectangle of width bearers around the barcode.

        quiet (bool, default 1):
            Whether to include quiet zones in the symbol.

        lquiet (float, see default below):
            Quiet zone size to left of code, if quiet is true.
            Default is the greater of .25 inch, or .15 times the symbol's
            length.

        rquiet (float, defaults as above):
            Quiet zone size to right left of code, if quiet is true.

        stop (bool, default 1):
            Whether to include start/stop symbols.

    Sources of Information on Interleaved 2 of 5:

    http://www.semiconductor.agilent.com/barcode/sg/Misc/i_25.html
    http://www.adams1.com/pub/russadam/i25code.html

    Official Spec, "ANSI/AIM BC2-1995, USS" is available for US$45 from
    http://www.aimglobal.org/aimstore/
    """
    patterns = {'start': 'bsbs', 'stop': 'Bsb', 'B0': 'bbBBb', 'S0': 'ssSSs', 'B1': 'BbbbB', 'S1': 'SsssS', 'B2': 'bBbbB', 'S2': 'sSssS', 'B3': 'BBbbb', 'S3': 'SSsss', 'B4': 'bbBbB', 'S4': 'ssSsS', 'B5': 'BbBbb', 'S5': 'SsSss', 'B6': 'bBBbb', 'S6': 'sSSss', 'B7': 'bbbBB', 'S7': 'sssSS', 'B8': 'BbbBb', 'S8': 'SssSs', 'B9': 'bBbBb', 'S9': 'sSsSs'}
    barHeight = None
    barWidth = inch * 0.0075
    ratio = 2.2
    checksum = 1
    bearers = 3.0
    bearerBox = False
    quiet = 1
    lquiet = None
    rquiet = None
    stop = 1

    def __init__(self, value='', **args):
        if type(value) == type(1):
            value = str(value)
        for k, v in args.items():
            setattr(self, k, v)
        if self.quiet:
            if self.lquiet is None:
                self.lquiet = min(inch * 0.25, self.barWidth * 10.0)
                self.rquiet = min(inch * 0.25, self.barWidth * 10.0)
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
        cs = self.checksum
        c = len(s)
        if c % 2 == 0 and cs or (c % 2 == 1 and (not cs)):
            s = '0' + s
            c += 1
        if cs:
            c = 3 * sum([int(s[i]) for i in range(0, c, 2)]) + sum([int(s[i]) for i in range(1, c, 2)])
            s += str((10 - c) % 10)
        self.encoded = s

    def decompose(self):
        dval = self.stop and [self.patterns['start']] or []
        a = dval.append
        for i in range(0, len(self.encoded), 2):
            b = self.patterns['B' + self.encoded[i]]
            s = self.patterns['S' + self.encoded[i + 1]]
            for i in range(0, len(b)):
                a(b[i] + s[i])
        if self.stop:
            a(self.patterns['stop'])
        self.decomposed = ''.join(dval)
        return self.decomposed