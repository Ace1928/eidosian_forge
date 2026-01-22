import operator
import os
import platform
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ._parser import (
from ._parser import (
from ._tokenizer import ParserSyntaxError
from .specifiers import InvalidSpecifier, Specifier
from .utils import canonicalize_name
def _eval_op(lhs: str, op: Op, rhs: str) -> bool:
    try:
        spec = Specifier(''.join([op.serialize(), rhs]))
    except InvalidSpecifier:
        pass
    else:
        return spec.contains(lhs, prereleases=True)
    oper: Optional[Operator] = _operators.get(op.serialize())
    if oper is None:
        raise UndefinedComparison(f'Undefined {op!r} on {lhs!r} and {rhs!r}.')
    return oper(lhs, rhs)