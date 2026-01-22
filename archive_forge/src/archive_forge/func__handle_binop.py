import ast
import builtins
import operator
from collections import ChainMap, OrderedDict, deque
from contextlib import suppress
from types import FrameType
from typing import Any, Tuple, Iterable, List, Mapping, Dict, Union, Set
from pure_eval.my_getattr_static import getattr_static
from pure_eval.utils import (
def _handle_binop(self, node):
    op_type = type(node.op)
    op = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.LShift: operator.lshift, ast.RShift: operator.rshift, ast.BitOr: operator.or_, ast.BitXor: operator.xor, ast.BitAnd: operator.and_}.get(op_type)
    if not op:
        raise CannotEval
    left = self[node.left]
    hash_type = is_any(type(left), set, frozenset, dict, OrderedDict)
    left = of_standard_types(left, check_dict_values=False, deep=hash_type)
    formatting = type(left) in (str, bytes) and op_type == ast.Mod
    right = of_standard_types(self[node.right], check_dict_values=formatting, deep=formatting or hash_type)
    try:
        return op(left, right)
    except Exception as e:
        raise CannotEval from e