from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
class PreScan(VisitorBase):

    def __init__(self, context: VisitorContext, *args: Any, **kwargs: Any):
        VisitorContext.__init__(self, context, *args, **kwargs)
        self._in_agg = False
        self._in_window = False
        self._non_agg_mentions = ColumnMentions()
        self._agg_funcs: List[ParserRuleContext] = []
        self._group_by_expressions: List[ParserRuleContext] = []
        self._window_funcs: List[ParserRuleContext] = []
        self._having_expressions: List[ParserRuleContext] = []
        self._not_in_agg_star: Any = None
        self._func_args: List[Tuple[ParserRuleContext, bool, bool]] = []

    def visitStar(self, ctx: qp.StarContext) -> None:
        if self.to_str(ctx) == '*':
            self.set('select_all', True)
        else:
            self.add_column_mentions(ctx)
        if not self._in_agg:
            self._not_in_agg_star = ctx

    def visitColumnReference(self, ctx: qp.ColumnReferenceContext) -> None:
        m = self.add_column_mentions(ctx)
        if not self._in_agg:
            self._non_agg_mentions.add(m)

    def visitDereference(self, ctx: qp.DereferenceContext) -> None:
        m = self.add_column_mentions(ctx)
        if not self._in_agg:
            self._non_agg_mentions.add(m)

    def visitAliasedQuery(self, ctx: qp.QueryContext) -> None:
        return

    def visitFirst(self, ctx: qp.FirstContext) -> None:
        self._handle_func_call('first', ctx)

    def visitLast(self, ctx: qp.LastContext) -> None:
        self._handle_func_call('last', ctx)

    def visitFunctionCall(self, ctx: qp.FunctionCallContext) -> None:
        func = self.visit(ctx.functionName())
        if ctx.windowSpec() is None:
            self._handle_func_call(func, ctx)
        else:
            self.set('has_window_func', True)
            self.assert_support(not self._in_agg and (not self._in_window), ctx)
            self._window_funcs.append(ctx)
            self._in_window = True
            self.visitChildren(ctx)
            self._in_window = False
        for a in ctx.argument:
            is_col, is_single = PreFunctionArgument(self).visit_argument(a)
            self._func_args.append((a, is_col, is_single))

    def visitPredicate(self, ctx: qp.PredicateContext):
        self.visitChildren(ctx)
        if ctx.expression() is not None:
            for a in ctx.expression():
                is_col, is_single = PreFunctionArgument(self).visit_argument(a)
                self._func_args.append((a, is_col, is_single))

    def visitWindowRef(self, ctx: qp.WindowRefContext) -> None:
        self.not_support(ctx)

    def visitWindowDef(self, ctx: qp.WindowDefContext) -> None:
        self.set('has_window', True)
        super().visitWindowDef(ctx)

    def visitAggregationClause(self, ctx: qp.AggregationClauseContext):
        for e in ctx.expression():
            self._in_agg = True
            self._group_by_expressions.append(e)
            self.visit(e)
            self._in_agg = False

    def visitRegularQuerySpecification(self, ctx: qp.RegularQuerySpecificationContext) -> None:
        self.assert_none(ctx.lateralView(), ctx.windowClause())
        if ctx.fromClause() is not None:
            self.copy(PreFrom).visit(ctx.fromClause())
        self.visit(ctx.selectClause())
        if ctx.whereClause() is not None:
            self.visit(ctx.whereClause())
        if ctx.aggregationClause() is not None:
            self.visit(ctx.aggregationClause())
        if ctx.havingClause() is not None:
            self.visit(ctx.havingClause())
        self.assert_support(not self.get('has_agg_func', False) or self._not_in_agg_star is None, self._not_in_agg_star)
        self.assert_support(not ((self.get('has_agg_func', False) or ctx.aggregationClause() is not None) and self.get('has_window_func', False)), "can't have both aggregation and window")
        if len(self.joins) > 0:
            joined: Set[str] = set([self.joins[0].df_name])
            for i in range(1, len(self.joins)):
                j = self.joins[i]
                right = j.df_name
                v = self.copy(JoinCriteria)
                v._joined, v._right_name = (joined, right)
                if j.criteria_context is not None:
                    j.set_conditions(v.visit(j.criteria_context))
                else:
                    j.set_conditions([])
                joined.add(right)
        self.set('non_agg_mentions', self._non_agg_mentions)
        self.set('agg_funcs', self._agg_funcs)
        self.set('group_by_expressions', self._group_by_expressions)
        self.set('window_funcs', self._window_funcs)
        self.set('func_arg_types', {id(x[0]): (x[1], x[2]) for x in self._func_args})

    def _handle_func_call(self, func: str, ctx: Any) -> None:
        if is_agg(func):
            self.set('has_agg_func', True)
            self.assert_support(not self._in_agg and (not self._in_window), ctx)
            self._agg_funcs.append(ctx)
            self._in_agg = True
        self.visitChildren(ctx)
        self._in_agg = False