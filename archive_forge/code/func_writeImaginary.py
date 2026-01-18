import collections
import warnings
from sympy.external import import_module
def writeImaginary(self, ctx):
    a = ', '.join(self.var_list) + ' = ' + '_sm.symbols(' + "'" + ' '.join(self.var_list) + "')\n"
    b = ', '.join(self.var_list) + ' = ' + '_sm.I\n'
    self.write(a)
    self.write(b)
    self.var_list = []