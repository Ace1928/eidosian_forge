import ast
import builtins
import operator
from collections import ChainMap, OrderedDict, deque
from contextlib import suppress
from types import FrameType
from typing import Any, Tuple, Iterable, List, Mapping, Dict, Union, Set
from pure_eval.my_getattr_static import getattr_static
from pure_eval.utils import (
def _handle_unary(self, node: ast.UnaryOp):
    value = of_standard_types(self[node.operand], check_dict_values=False, deep=False)
    op_type = type(node.op)
    op = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_, ast.Invert: operator.invert}[op_type]
    try:
        return op(value)
    except Exception as e:
        raise CannotEval from e