import ast
import builtins
import operator
from collections import ChainMap, OrderedDict, deque
from contextlib import suppress
from types import FrameType
from typing import Any, Tuple, Iterable, List, Mapping, Dict, Union, Set
from pure_eval.my_getattr_static import getattr_static
from pure_eval.utils import (
def _handle_boolop(self, node):
    left = of_standard_types(self[node.values[0]], check_dict_values=False, deep=False)
    for right in node.values[1:]:
        if isinstance(node.op, ast.Or):
            left = left or of_standard_types(self[right], check_dict_values=False, deep=False)
        else:
            assert isinstance(node.op, ast.And)
            left = left and of_standard_types(self[right], check_dict_values=False, deep=False)
    return left