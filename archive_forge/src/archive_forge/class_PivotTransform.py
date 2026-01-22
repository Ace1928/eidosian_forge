from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class PivotTransform(Transform):
    """PivotTransform schema wrapper

    Parameters
    ----------

    pivot : str, :class:`FieldName`
        The data field to pivot on. The unique values of this field become new field names
        in the output stream.
    value : str, :class:`FieldName`
        The data field to populate pivoted fields. The aggregate values of this field become
        the values of the new pivoted fields.
    groupby : Sequence[str, :class:`FieldName`]
        The optional data fields to group by. If not specified, a single group containing
        all data objects will be used.
    limit : float
        An optional parameter indicating the maximum number of pivoted fields to generate.
        The default ( ``0`` ) applies no limit. The pivoted ``pivot`` names are sorted in
        ascending order prior to enforcing the limit. **Default value:** ``0``
    op : :class:`AggregateOp`, Literal['argmax', 'argmin', 'average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb']
        The aggregation operation to apply to grouped ``value`` field values. **Default
        value:** ``sum``
    """
    _schema = {'$ref': '#/definitions/PivotTransform'}

    def __init__(self, pivot: Union[str, 'SchemaBase', UndefinedType]=Undefined, value: Union[str, 'SchemaBase', UndefinedType]=Undefined, groupby: Union[Sequence[Union[str, 'SchemaBase']], UndefinedType]=Undefined, limit: Union[float, UndefinedType]=Undefined, op: Union['SchemaBase', Literal['argmax', 'argmin', 'average', 'count', 'distinct', 'max', 'mean', 'median', 'min', 'missing', 'product', 'q1', 'q3', 'ci0', 'ci1', 'stderr', 'stdev', 'stdevp', 'sum', 'valid', 'values', 'variance', 'variancep', 'exponential', 'exponentialb'], UndefinedType]=Undefined, **kwds):
        super(PivotTransform, self).__init__(pivot=pivot, value=value, groupby=groupby, limit=limit, op=op, **kwds)