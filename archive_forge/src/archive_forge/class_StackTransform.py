from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class StackTransform(Transform):
    """StackTransform schema wrapper

    Parameters
    ----------

    groupby : Sequence[str, :class:`FieldName`]
        The data fields to group by.
    stack : str, :class:`FieldName`
        The field which is stacked.
    offset : Literal['zero', 'center', 'normalize']
        Mode for stacking marks. One of ``"zero"`` (default), ``"center"``, or
        ``"normalize"``. The ``"zero"`` offset will stack starting at ``0``. The
        ``"center"`` offset will center the stacks. The ``"normalize"`` offset will compute
        percentage values for each stack point, with output values in the range ``[0,1]``.

        **Default value:** ``"zero"``
    sort : Sequence[dict, :class:`SortField`]
        Field that determines the order of leaves in the stacked charts.
    as : str, :class:`FieldName`, Sequence[str, :class:`FieldName`]
        Output field names. This can be either a string or an array of strings with two
        elements denoting the name for the fields for stack start and stack end
        respectively. If a single string(e.g., ``"val"`` ) is provided, the end field will
        be ``"val_end"``.
    """
    _schema = {'$ref': '#/definitions/StackTransform'}

    def __init__(self, groupby: Union[Sequence[Union[str, 'SchemaBase']], UndefinedType]=Undefined, stack: Union[str, 'SchemaBase', UndefinedType]=Undefined, offset: Union[Literal['zero', 'center', 'normalize'], UndefinedType]=Undefined, sort: Union[Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, **kwds):
        super(StackTransform, self).__init__(groupby=groupby, stack=stack, offset=offset, sort=sort, **kwds)