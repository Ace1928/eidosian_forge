from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
class ExpressionVisitor(VisitorBase):

    def __init__(self, context: VisitorContext):
        super().__init__(context)

    def visitStar(self, ctx: qp.StarContext) -> List[Column]:
        return self._get_columns(ctx)

    def visitColumnReference(self, ctx: qp.ColumnReferenceContext) -> List[Column]:
        return self._get_columns(ctx)

    def visitDereference(self, ctx: qp.DereferenceContext) -> List[Column]:
        return self._get_columns(ctx)

    def visitConstantDefault(self, ctx: qp.ConstantDefaultContext) -> List[Column]:
        raw = self.to_str(ctx, '')
        if raw.lower() == 'true':
            value = True
        elif raw.lower() == 'false':
            value = False
        else:
            value = eval(raw)
        return [self.workflow.const_to_col(value)]

    def visitParenthesizedExpression(self, ctx: qp.ParenthesizedExpressionContext) -> List[Column]:
        return self.visit(ctx.expression())

    def visitArithmeticUnary(self, ctx: qp.ArithmeticUnaryContext) -> List[Column]:
        op = self.to_str(ctx.operator)
        v = self._get_single_column(ctx.valueExpression())
        return [self.workflow.op_to_col('basic_unary_arithmetic_op', v, op)]

    def visitArithmeticBinary(self, ctx: qp.ArithmeticBinaryContext) -> List[Column]:
        op = self.to_str(ctx.operator)
        left, right = self._get_single_left_right(ctx)
        return [self.workflow.op_to_col('binary_arithmetic_op', left, right, op)]

    def visitLogicalNot(self, ctx: qp.LogicalNotContext) -> List[Column]:
        col = self._get_single_column(ctx.booleanExpression())
        return [self.workflow.op_to_col('logical_not', col)]

    def visitPredicated(self, ctx: qp.PredicatedContext) -> List[Column]:
        if ctx.predicate() is None:
            return self.visit(ctx.valueExpression())
        col = self._get_single_column(ctx.valueExpression())
        kind = self.to_str(ctx.predicate().kind).lower()
        positive = ctx.predicate().NOT() is None
        if kind in ['null', 'true', 'false']:
            vs = IsValueSpec(kind, positive)
            return [self.workflow.op_to_col('is_value', col, vs)]
        if kind == 'in':
            self.assert_support(len(ctx.predicate().expression()) > 0, ctx)
            in_cols: List[Column] = []
            in_values: List[Any] = []
            for e in ctx.predicate().expression():
                is_col, _ = self.get('func_arg_types')[id(e)]
                if not is_col:
                    in_values.append(eval(self.to_str(e)))
                else:
                    in_cols.append(self._get_single_column(e))
            return [self.workflow.op_to_col('is_in', col, *in_cols, *in_values, positive=positive)]
        if kind == 'between':
            lower = self._get_single_column(ctx.predicate().lower)
            upper = self._get_single_column(ctx.predicate().upper)
            return [self.workflow.op_to_col('is_between', col, lower, upper, positive=positive)]
        return self.not_support(ctx)

    def visitComparison(self, ctx: qp.ComparisonContext) -> List[Column]:

        def to_op(o: qp.ComparisonOperatorContext):
            if o.EQ():
                return '=='
            if o.NEQ() or o.NEQJ():
                return '!='
            if o.LT():
                return '<'
            if o.LTE():
                return '<='
            if o.GT():
                return '>'
            if o.GTE():
                return '>='
            self.not_support('comparator ' + self.to_str(o))
        op = to_op(ctx.comparisonOperator())
        left, right = self._get_single_left_right(ctx)
        return [self.workflow.op_to_col('comparison_op', left, right, op)]

    def visitLogicalBinary(self, ctx: qp.LogicalBinaryContext) -> List[Column]:
        op = self.to_str(ctx.operator).lower()
        left, right = self._get_single_left_right(ctx)
        return [self.workflow.op_to_col('binary_logical_op', left, right, op)]

    def visitFirst(self, ctx: qp.FirstContext) -> List[Column]:
        return [self.current[self.get('agg_func_to_col')[id(ctx)]]]

    def visitLast(self, ctx: qp.LastContext) -> List[Column]:
        return [self.current[self.get('agg_func_to_col')[id(ctx)]]]

    def visitFunctionCall(self, ctx: qp.FunctionCallContext) -> List[Column]:
        if ctx.windowSpec() is not None:
            return [self.current[self.get('window_func_to_col')[id(ctx)]]]
        func = self.visit(ctx.functionName())
        if is_agg(func):
            return [self.current[self.get('agg_func_to_col')[id(ctx)]]]
        self.not_support(ctx)

    def visitSearchedCase(self, ctx: qp.SearchedCaseContext) -> List[Column]:
        cols: List[Column] = []
        for when in ctx.whenClause():
            cols += self.visitWhenClause(when)
        cols.append(self._get_single_column(ctx.elseExpression))
        return [self.workflow.op_to_col('case_when', *cols)]

    def visitWhenClause(self, ctx: qp.WhenClauseContext) -> Tuple[Column, Column]:
        return (self._get_single_column(ctx.condition), self._get_single_column(ctx.result))

    def _get_single_left_right(self, ctx: Any) -> List[Column]:
        left = self._get_single_column(ctx.left)
        right = self._get_single_column(ctx.right)
        return [left, right]

    def _get_single_column(self, ctx: Any) -> Column:
        c = list(self.visit(ctx))
        self.assert_support(len(c) == 1, ctx)
        return c[0]

    def _get_columns(self, ctx: Any) -> List[Column]:
        cols = self.get_column_mentions(ctx)
        return [self.current[c.encoded] for c in cols]