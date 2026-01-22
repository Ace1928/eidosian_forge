from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
class SelectVisitor(ExpressionVisitor):

    def __init__(self, context: VisitorContext):
        super().__init__(context)

    def visitNamedExpression(self, ctx: qp.NamedExpressionContext) -> Iterable[Column]:
        self.assert_none(ctx.identifierList())
        alias = ''
        if ctx.errorCapturingIdentifier() is not None:
            alias = self.to_str(ctx.errorCapturingIdentifier(), '', unquote=True)
        for col in self.visit(ctx.expression()):
            yield (col if alias == '' else col.rename(alias))

    def visitNamedExpressionSeq(self, ctx: qp.NamedExpressionSeqContext) -> Iterable[Column]:
        for ne in ctx.namedExpression():
            for r in self.visit(ne):
                yield r

    def visitSelectClause(self, ctx: qp.SelectClauseContext) -> None:
        self.update_current(WorkflowDataFrame(*list(self.visit(ctx.namedExpressionSeq()))))
        if ctx.setQuantifier() is not None and ctx.setQuantifier().DISTINCT() is not None:
            self.update_current(self.workflow.op_to_df(list(self.current.keys()), 'drop_duplicates', self.current))

    def _get_columns(self, ctx: Any) -> List[Column]:
        if self.to_str(ctx) == '*':
            cols: List[Column] = []
            for c in self.current.values():
                cn = self.get('encoded_map', None).get(c.name, '')
                if cn != '':
                    cols.append(c.rename(cn))
            return cols
        return [self.current[x.encoded].rename(x.col_name) for x in self.get_column_mentions(ctx)]