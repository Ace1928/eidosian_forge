from NumPy.
import os
from pathlib import Path
import ast
import tokenize
import scipy
import pytest
class ParseCall(ast.NodeVisitor):

    def __init__(self):
        self.ls = []

    def visit_Attribute(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        self.ls.append(node.attr)

    def visit_Name(self, node):
        self.ls.append(node.id)