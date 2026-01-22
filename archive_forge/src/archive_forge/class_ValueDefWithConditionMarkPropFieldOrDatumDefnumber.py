from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class ValueDefWithConditionMarkPropFieldOrDatumDefnumber(MarkPropDefnumber, NumericMarkPropDef):
    """ValueDefWithConditionMarkPropFieldOrDatumDefnumber schema wrapper

    Parameters
    ----------

    condition : dict, :class:`ConditionalValueDefnumberExprRef`, :class:`ConditionalMarkPropFieldOrDatumDef`, :class:`ConditionalParameterValueDefnumberExprRef`, :class:`ConditionalPredicateValueDefnumberExprRef`, :class:`ConditionalParameterMarkPropFieldOrDatumDef`, :class:`ConditionalPredicateMarkPropFieldOrDatumDef`, Sequence[dict, :class:`ConditionalValueDefnumberExprRef`, :class:`ConditionalParameterValueDefnumberExprRef`, :class:`ConditionalPredicateValueDefnumberExprRef`]
        A field definition or one or more value definition(s) with a parameter predicate.
    value : dict, float, :class:`ExprRef`
        A constant value in visual domain (e.g., ``"red"`` / ``"#0099ff"`` / `gradient
        definition <https://vega.github.io/vega-lite/docs/types.html#gradient>`__ for color,
        values between ``0`` to ``1`` for opacity).
    """
    _schema = {'$ref': '#/definitions/ValueDefWithCondition<MarkPropFieldOrDatumDef,number>'}

    def __init__(self, condition: Union[dict, 'SchemaBase', Sequence[Union[dict, 'SchemaBase']], UndefinedType]=Undefined, value: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(ValueDefWithConditionMarkPropFieldOrDatumDefnumber, self).__init__(condition=condition, value=value, **kwds)