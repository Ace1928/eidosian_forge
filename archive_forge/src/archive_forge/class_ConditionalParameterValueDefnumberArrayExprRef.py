from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ConditionalParameterValueDefnumberArrayExprRef(ConditionalValueDefnumberArrayExprRef):
    """ConditionalParameterValueDefnumberArrayExprRef schema wrapper

    Parameters
    ----------

    param : str, :class:`ParameterName`
        Filter using a parameter name.
    value : dict, Sequence[float], :class:`ExprRef`
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    empty : bool
        For selection parameters, the predicate of empty selections returns true by default.
        Override this behavior, by setting this property ``empty: false``.
    """
    _schema = {'$ref': '#/definitions/ConditionalParameter<ValueDef<(number[]|ExprRef)>>'}

    def __init__(self, param: Union[str, 'SchemaBase', UndefinedType]=Undefined, value: Union[dict, '_Parameter', 'SchemaBase', Sequence[float], UndefinedType]=Undefined, empty: Union[bool, UndefinedType]=Undefined, **kwds):
        super(ConditionalParameterValueDefnumberArrayExprRef, self).__init__(param=param, value=value, empty=empty, **kwds)