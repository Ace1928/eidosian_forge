from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
def _get_internal_col(self, ctx: Any) -> Tuple[str, Column]:
    internal_expr = self._obj_to_col_name(self.expr(ctx))
    if internal_expr not in self._internal_exprs:
        col = ExpressionVisitor(self)._get_single_column(ctx)
        self._internal_exprs[internal_expr] = col.rename(internal_expr)
    return (internal_expr, self._internal_exprs[internal_expr])