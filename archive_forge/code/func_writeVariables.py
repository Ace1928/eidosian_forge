import collections
import warnings
from sympy.external import import_module
def writeVariables(self, ctx):
    if self.var_list:
        for i in range(self.maxDegree + 1):
            if i == 0:
                j = ''
                t = ''
            else:
                j = str(i)
                t = ', '
            l = []
            for k in list(filter(lambda x: self.sign[x] == i, self.var_list)):
                if i == 0:
                    l.append(k)
                if i == 1:
                    l.append(k[:-1])
                if i > 1:
                    l.append(k[:-2])
            a = ', '.join(list(filter(lambda x: self.sign[x] == i, self.var_list))) + ' = ' + '_me.dynamicsymbols(' + "'" + ' '.join(l) + "'" + t + j + ')\n'
            l = []
            self.write(a)
        self.maxDegree = 0
    self.var_list = []