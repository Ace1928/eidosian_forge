import string
import numpy as np
from ase.io import string2index
from ase.io.formats import parse_filename
from ase.data import chemical_symbols
class TableFormat:

    def __init__(self, columnwidth=9, precision=2, representation='E', toprule='=', midrule='-', bottomrule='='):
        self.precision = precision
        self.representation = representation
        self.columnwidth = columnwidth
        self.formatter = MapFormatter().format
        self.toprule = toprule
        self.midrule = midrule
        self.bottomrule = bottomrule
        self.fmt_class = {'signed float': '{{: ^{}.{}{}}}'.format(self.columnwidth, self.precision - 1, self.representation), 'unsigned float': '{{:^{}.{}{}}}'.format(self.columnwidth, self.precision - 1, self.representation), 'int': '{{:^{}n}}'.format(self.columnwidth), 'str': '{{:^{}s}}'.format(self.columnwidth), 'conv': '{{:^{}h}}'.format(self.columnwidth)}
        fmt = {}
        signed_floats = ['dx', 'dy', 'dz', 'dfx', 'dfy', 'dfz', 'afx', 'afy', 'afz', 'p1x', 'p2x', 'p1y', 'p2y', 'p1z', 'p2z', 'f1x', 'f2x', 'f1y', 'f2y', 'f1z', 'f2z']
        for sf in signed_floats:
            fmt[sf] = self.fmt_class['signed float']
        unsigned_floats = ['d', 'df', 'af', 'p1', 'p2', 'f1', 'f2']
        for usf in unsigned_floats:
            fmt[usf] = self.fmt_class['unsigned float']
        integers = ['i', 'an', 't'] + ['r' + sf for sf in signed_floats] + ['r' + usf for usf in unsigned_floats]
        for i in integers:
            fmt[i] = self.fmt_class['int']
        fmt['el'] = self.fmt_class['conv']
        self.fmt = fmt