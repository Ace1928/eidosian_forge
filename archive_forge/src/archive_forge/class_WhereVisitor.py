from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
class WhereVisitor(ExpressionVisitor):

    def __init__(self, context: VisitorContext):
        super().__init__(context)

    def visitWhereClause(self, ctx: qp.WhereClauseContext) -> None:
        self._filter(ctx)

    def visitHavingClause(self, ctx: qp.HavingClauseContext):
        self._filter(ctx)

    def _filter(self, ctx: Any):
        cond = self._get_single_column(ctx.booleanExpression())
        current = self.workflow.op_to_df(list(self.current.keys()), 'filter_df', self.current, cond)
        self.update_current(current)