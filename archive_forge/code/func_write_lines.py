import unittest
from Cython import StringIOTree as stringtree
def write_lines(self, linenos, tree=None):
    for lineno in linenos:
        self.write_line(lineno, tree=tree)