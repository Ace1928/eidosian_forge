from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
def visit_argument(self, ctx: Any) -> Tuple[bool, bool]:
    self._has_star = False
    self._has_func = False
    self._cols = 0
    self._constants = 0
    self.visit(ctx)
    is_col = self._cols > 0
    self.assert_support(is_col or self._constants > 0, ctx)
    if is_col:
        is_single = self._cols == 1 and (not self._has_star) and (not self._has_func) and (self._constants == 0)
    else:
        is_single = self._constants == 1 and (not self._has_func)
    return (is_col, is_single)