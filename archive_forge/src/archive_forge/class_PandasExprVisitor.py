from __future__ import annotations
import ast
from functools import (
from keyword import iskeyword
import tokenize
from typing import (
import numpy as np
from pandas.errors import UndefinedVariableError
import pandas.core.common as com
from pandas.core.computation.ops import (
from pandas.core.computation.parsing import (
from pandas.core.computation.scope import Scope
from pandas.io.formats import printing
@disallow((_unsupported_nodes | _python_not_supported) - (_boolop_nodes | frozenset(['BoolOp', 'Attribute', 'In', 'NotIn', 'Tuple'])))
class PandasExprVisitor(BaseExprVisitor):

    def __init__(self, env, engine, parser, preparser=partial(_preparse, f=_compose(_replace_locals, _replace_booleans, clean_backtick_quoted_toks))) -> None:
        super().__init__(env, engine, parser, preparser)