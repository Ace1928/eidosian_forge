import collections
import warnings
from sympy.external import import_module
def writeConstants(self, ctx):
    l1 = list(filter(lambda x: self.sign[x] == 'o', self.var_list))
    l2 = list(filter(lambda x: self.sign[x] == '+', self.var_list))
    l3 = list(filter(lambda x: self.sign[x] == '-', self.var_list))
    try:
        if self.settings['complex'] == 'on':
            real = ', real=True'
        elif self.settings['complex'] == 'off':
            real = ''
    except Exception:
        real = ', real=True'
    if l1:
        a = ', '.join(l1) + ' = ' + '_sm.symbols(' + "'" + ' '.join(l1) + "'" + real + ')\n'
        self.write(a)
    if l2:
        a = ', '.join(l2) + ' = ' + '_sm.symbols(' + "'" + ' '.join(l2) + "'" + real + ', nonnegative=True)\n'
        self.write(a)
    if l3:
        a = ', '.join(l3) + ' = ' + '_sm.symbols(' + "'" + ' '.join(l3) + "'" + real + ', nonpositive=True)\n'
        self.write(a)
    self.var_list = []