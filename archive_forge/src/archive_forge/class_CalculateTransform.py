from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class CalculateTransform(Transform):
    """CalculateTransform schema wrapper

    Parameters
    ----------

    calculate : str
        A `expression <https://vega.github.io/vega-lite/docs/types.html#expression>`__
        string. Use the variable ``datum`` to refer to the current data object.
    as : str, :class:`FieldName`
        The field for storing the computed formula value.
    """
    _schema = {'$ref': '#/definitions/CalculateTransform'}

    def __init__(self, calculate: Union[str, UndefinedType]=Undefined, **kwds):
        super(CalculateTransform, self).__init__(calculate=calculate, **kwds)