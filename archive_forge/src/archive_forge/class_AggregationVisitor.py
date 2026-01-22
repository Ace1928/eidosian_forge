from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
class AggregationVisitor(VisitorBase):

    def __init__(self, context: VisitorContext):
        super().__init__(context)
        self._internal_exprs: Dict[str, Column] = {}
        self._gp_map: Dict[str, Tuple[str, AggFunctionSpec]] = {}
        self._gp_keys: List[str] = []
        self._f_to_name: Dict[int, str] = {}

    def visitFirst(self, ctx: qp.FirstContext) -> Tuple[AggFunctionSpec, List[Any]]:
        args = [ctx.expression()]
        dropna = ctx.IGNORE() is not None
        func = AggFunctionSpec('first', unique=False, dropna=dropna)
        return (func, args)

    def visitLast(self, ctx: qp.LastContext) -> Tuple[AggFunctionSpec, List[Any]]:
        args = [ctx.expression()]
        dropna = ctx.IGNORE() is not None
        func = AggFunctionSpec('last', unique=False, dropna=dropna)
        return (func, args)

    def visitFunctionCall(self, ctx: qp.FunctionCallContext) -> Tuple[AggFunctionSpec, List[Any]]:
        func_name = self.visitFunctionName(ctx.functionName())
        args = ctx.argument
        unique = ctx.setQuantifier() is not None and ctx.setQuantifier().DISTINCT() is not None
        func = AggFunctionSpec(func_name, unique=unique, dropna=func_name not in ['first', 'last', 'first_value', 'last_value'])
        return (func, args)

    def visitRegularQuerySpecification(self, ctx: qp.RegularQuerySpecificationContext) -> None:
        self._handle_mentions()
        self._handle_agg_funcs()
        self._handle_group_by()
        if len(self._internal_exprs) > 0:
            self.update_current(WorkflowDataFrame(self.current, *list(self._internal_exprs.values())))
        self.update_current(self.workflow.op_to_df(list(self._gp_map.keys()), 'group_agg', self.current, self._gp_keys, self._gp_map))
        self.set('agg_func_to_col', self._f_to_name)

    def _handle_mentions(self) -> None:
        for m in self.get('non_agg_mentions'):
            func = AggFunctionSpec('first', unique=False, dropna=False)
            self._gp_map[m.encoded] = (m.encoded, func)

    def _handle_agg_funcs(self) -> None:
        for f_ctx in self.get('agg_funcs', None):
            func, args = self.visit(f_ctx)
            name = self._obj_to_col_name(self.expr(f_ctx))
            if func.name != 'count':
                self.assert_support(len(args) == 1, f_ctx)
                internal_expr = self._get_func_args(args)
            else:
                if len(args) == 0:
                    expr = '*'
                else:
                    expr = self.expr(args[0])
                if expr == '*':
                    self.assert_support(len(args) <= 1, f_ctx)
                    internal_expr = ['*']
                    func = AggFunctionSpec(func.name, func.unique, dropna=False)
                else:
                    internal_expr = self._get_func_args(args)
                    func = AggFunctionSpec(func.name, func.unique, dropna=True)
            self._gp_map[name] = (','.join(internal_expr), func)
            self._f_to_name[id(f_ctx)] = name

    def _handle_group_by(self) -> None:
        for i_ctx in self.get('group_by_expressions'):
            internal_expr, _ = self._get_internal_col(i_ctx)
            self._gp_keys.append(internal_expr)

    def _get_func_args(self, args: Any) -> List[str]:
        e: List[str] = []
        for i in range(len(args)):
            x, _ = self._get_internal_col(args[i])
            e.append(x)
        return e

    def _get_internal_col(self, ctx: Any) -> Tuple[str, Column]:
        internal_expr = self._obj_to_col_name(self.expr(ctx))
        if internal_expr not in self._internal_exprs:
            col = ExpressionVisitor(self)._get_single_column(ctx)
            self._internal_exprs[internal_expr] = col.rename(internal_expr)
        return (internal_expr, self._internal_exprs[internal_expr])