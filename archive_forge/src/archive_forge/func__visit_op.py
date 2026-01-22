import sys
from typing import Dict, List, Optional, Type, overload
def _visit_op(self, node: ast.AST) -> str:
    return OPERATORS[node.__class__]