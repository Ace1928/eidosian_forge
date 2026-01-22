from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
class QuantileAggregate(CompoundAggregateWithColArg):
    """
        A QUANTILE aggregate generator.

        Parameters
        ----------
        builder : CalciteBuilder
            A builder to use for translation.
        arg : List of BaseExpr
            A list of 3 values:
                0. InputRefExpr - the column to compute the quantiles for.
                1. LiteralExpr - the quantile value.
                2. str - the interpolation method to use.
        """

    def __init__(self, builder, arg):
        super().__init__('QUANTILE', builder, arg, _quantile_agg_dtype(arg[0]._dtype))
        self._interpolation = arg[2].val.upper()

    def gen_agg_exprs(self):
        exprs = super().gen_agg_exprs()
        for expr in exprs.values():
            expr.interpolation = self._interpolation
        return exprs