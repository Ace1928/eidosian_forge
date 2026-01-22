from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
class CompoundAggregateWithColArg(CompoundAggregate):
    """
        A base class for a compound aggregate that require a `LiteralExpr` column argument.

        This aggregate requires 2 arguments. The first argument is an `InputRefExpr`,
        refering to the aggregation column. The second argument is a `LiteralExpr`,
        this expression is added into the frame as a new column.

        Parameters
        ----------
        agg : str
            Aggregate name.
        builder : CalciteBuilder
            A builder to use for translation.
        arg : List of BaseExpr
            Aggregate arguments.
        dtype : dtype, optional
            Aggregate data type. If not specified, `_dtype` from the first argument is used.
        """

    def __init__(self, agg, builder, arg, dtype=None):
        assert isinstance(arg[0], InputRefExpr)
        assert isinstance(arg[1], LiteralExpr)
        super().__init__(builder, arg)
        self._agg = agg
        self._agg_column = f'{arg[0].column}__{agg}__'
        self._dtype = dtype or arg[0]._dtype

    def gen_proj_exprs(self):
        return {self._agg_column: self._arg[1]}

    def gen_agg_exprs(self):
        frame = self._arg[0].modin_frame
        return {self._agg_column: AggregateExpr(self._agg, [self._builder._ref_idx(frame, self._arg[0].column), self._builder._ref_idx(frame, self._agg_column)], dtype=self._dtype)}

    def gen_reduce_expr(self):
        return self._builder._ref(self._arg[0].modin_frame, self._agg_column)