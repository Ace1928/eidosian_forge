from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class JoinAggregateTransform(Transform):
    """JoinAggregateTransform schema wrapper

    Parameters
    ----------

    joinaggregate : Sequence[dict, :class:`JoinAggregateFieldDef`]
        The definition of the fields in the join aggregate, and what calculations to use.
    groupby : Sequence[str, :class:`FieldName`]
        The data fields for partitioning the data objects into separate groups. If
        unspecified, all data points will be in a single group.
    """
    _schema = {'$ref': '#/definitions/JoinAggregateTransform'}

    def __init__(self, joinaggregate: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, groupby: Union[Sequence[Union[str, 'SchemaBase']], UndefinedType]=Undefined, **kwds):
        super(JoinAggregateTransform, self).__init__(joinaggregate=joinaggregate, groupby=groupby, **kwds)