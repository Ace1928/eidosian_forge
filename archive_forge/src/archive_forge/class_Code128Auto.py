from reportlab.lib.units import inch
from reportlab.lib.utils import asNative
from reportlab.graphics.barcode.common import MultiWidthBarcode
from string import digits
class Code128Auto(Code128):
    """contributed by https://bitbucket.org/kylemacfarlane/
    see https://bitbucket.org/rptlab/reportlab/issues/69/implementations-of-code-128-auto-and-data
    """

    def encode(self):
        s = self.validated
        current_set = None
        l = []
        value = list(s)
        while value:
            c = value.pop(0)
            if c in digits and value and (value[0] in digits):
                c += value.pop(0)
            if c in setc:
                set_ = 'C'
            elif c in setb:
                set_ = 'B'
            else:
                set_ = 'A'
            if current_set != set_:
                if current_set:
                    l.append('TO_' + set_)
                else:
                    l.append('START_' + set_)
                current_set = set_
            l.append(c)
        l.append('STOP')
        start, set, shset = setmap[l[0]]
        e = [start]
        l = l[1:-1]
        while l:
            c = l[0]
            if c == 'SHIFT':
                e = e + [set[c], shset[l[1]]]
                l = l[2:]
            elif c in tos:
                e.append(set[c])
                set, shset = setmap[c]
                l = l[1:]
            else:
                e.append(set[c])
                l = l[1:]
        c = e[0]
        for i in range(1, len(e)):
            c = c + i * e[i]
        self.encoded = e + [c % 103, stop]
        return self.encoded