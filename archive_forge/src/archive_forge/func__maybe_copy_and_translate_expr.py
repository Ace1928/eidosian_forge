from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _maybe_copy_and_translate_expr(self, expr, ref_idx=False):
    """
            Translate an expression.

            Translate an expression replacing ``InputRefExpr`` with ``CalciteInputRefExpr``
            and ``CalciteInputIdxExpr``. An expression tree branches with input columns
            are copied into a new tree, other branches are used as is.

            Parameters
            ----------
            expr : BaseExpr
                An expression to translate.
            ref_idx : bool, default: False
                If True then translate ``InputRefExpr`` to ``CalciteInputIdxExpr``,
                use ``CalciteInputRefExr`` otherwise.

            Returns
            -------
            BaseExpr
                Translated expression.
            """
    if isinstance(expr, InputRefExpr):
        if ref_idx:
            return self.ref_idx(expr.modin_frame, expr.column)
        else:
            return self.ref(expr.modin_frame, expr.column)
    if isinstance(expr, AggregateExpr):
        expr = expr.copy()
        if expr.agg in self._no_arg_aggregates:
            expr.operands = []
        else:
            expr.operands[0] = self._maybe_copy_and_translate_expr(expr.operands[0], True)
        expr.agg = self._simple_aggregates[expr.agg]
        return expr
    gen = expr.nested_expressions()
    for op in gen:
        expr = gen.send(self._maybe_copy_and_translate_expr(op))
    return expr