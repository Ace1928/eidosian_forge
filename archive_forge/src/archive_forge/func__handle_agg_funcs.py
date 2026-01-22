from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
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