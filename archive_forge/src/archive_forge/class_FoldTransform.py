from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class FoldTransform(Transform):
    """FoldTransform schema wrapper

    Parameters
    ----------

    fold : Sequence[str, :class:`FieldName`]
        An array of data fields indicating the properties to fold.
    as : Sequence[str, :class:`FieldName`]
        The output field names for the key and value properties produced by the fold
        transform. **Default value:** ``["key", "value"]``
    """
    _schema = {'$ref': '#/definitions/FoldTransform'}

    def __init__(self, fold: Union[Sequence[Union[str, 'SchemaBase']], UndefinedType]=Undefined, **kwds):
        super(FoldTransform, self).__init__(fold=fold, **kwds)