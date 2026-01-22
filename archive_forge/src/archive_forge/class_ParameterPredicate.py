from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ParameterPredicate(Predicate):
    """ParameterPredicate schema wrapper

    Parameters
    ----------

    param : str, :class:`ParameterName`
        Filter using a parameter name.
    empty : bool
        For selection parameters, the predicate of empty selections returns true by default.
        Override this behavior, by setting this property ``empty: false``.
    """
    _schema = {'$ref': '#/definitions/ParameterPredicate'}

    def __init__(self, param: Union[str, 'SchemaBase', UndefinedType]=Undefined, empty: Union[bool, UndefinedType]=Undefined, **kwds):
        super(ParameterPredicate, self).__init__(param=param, empty=empty, **kwds)