from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class LookupSelection(VegaLiteSchema):
    """LookupSelection schema wrapper

    Parameters
    ----------

    key : str, :class:`FieldName`
        Key in data to lookup.
    param : str, :class:`ParameterName`
        Selection parameter name to look up.
    fields : Sequence[str, :class:`FieldName`]
        Fields in foreign data or selection to lookup. If not specified, the entire object
        is queried.
    """
    _schema = {'$ref': '#/definitions/LookupSelection'}

    def __init__(self, key: Union[str, 'SchemaBase', UndefinedType]=Undefined, param: Union[str, 'SchemaBase', UndefinedType]=Undefined, fields: Union[Sequence[Union[str, 'SchemaBase']], UndefinedType]=Undefined, **kwds):
        super(LookupSelection, self).__init__(key=key, param=param, fields=fields, **kwds)