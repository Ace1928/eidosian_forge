import ast
import sys
import importlib.util
def visit_AsyncFunctionDef(self, node):
    self.visit_FunctionDef(node, is_async=True)