from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class EncodingSortField(Sort):
    """EncodingSortField schema wrapper
    A sort definition for sorting a discrete scale in an encoding field definition.

    Parameters
    ----------

    field : str, dict, :class:`Field`, :class:`FieldName`, :class:`RepeatRef`
        The data `field <https://vega.github.io/vega-lite/docs/field.html>`__ to sort by.

        **Default value:** If unspecified, defaults to the field specified in the outer data
        reference.
    op : :class:`NonArgAggregateOp`, Literal['average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb']
        An `aggregate operation
        <https://vega.github.io/vega-lite/docs/aggregate.html#ops>`__ to perform on the
        field prior to sorting (e.g., ``"count"``, ``"mean"`` and ``"median"`` ). An
        aggregation is required when there are multiple values of the sort field for each
        encoded data field. The input data objects will be aggregated, grouped by the
        encoded data field.

        For a full list of operations, please see the documentation for `aggregate
        <https://vega.github.io/vega-lite/docs/aggregate.html#ops>`__.

        **Default value:** ``"sum"`` for stacked plots. Otherwise, ``"min"``.
    order : None, :class:`SortOrder`, Literal['ascending', 'descending']
        The sort order. One of ``"ascending"`` (default), ``"descending"``, or ``null`` (no
        not sort).
    """
    _schema = {'$ref': '#/definitions/EncodingSortField'}

    def __init__(self, field: Union[str, dict, 'SchemaBase', UndefinedType]=Undefined, op: Union['SchemaBase', Literal['average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb'], UndefinedType]=Undefined, order: Union[None, 'SchemaBase', Literal['ascending', 'descending'], UndefinedType]=Undefined, **kwds):
        super(EncodingSortField, self).__init__(field=field, op=op, order=order, **kwds)