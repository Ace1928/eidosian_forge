import re
class NumbaComplexPrinter:

    def __init__(self, val):
        self.val = val

    def to_string(self):
        return '%s+%sj' % (self.val['real'], self.val['imag'])