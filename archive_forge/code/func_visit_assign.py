from pythran.passmanager import Transformation
from pythran.tables import MODULES, pythran_ward
from pythran.syntax import PythranSyntaxError
import gast as ast
import logging
import os
def visit_assign(self, node):
    self.visit(node.value)
    for target in node.targets:
        self.visit(target)
    return node