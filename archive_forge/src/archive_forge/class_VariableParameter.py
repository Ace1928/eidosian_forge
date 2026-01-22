from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class VariableParameter(TopLevelParameter):
    """VariableParameter schema wrapper

    Parameters
    ----------

    name : str, :class:`ParameterName`
        A unique name for the variable parameter. Parameter names should be valid JavaScript
        identifiers: they should contain only alphanumeric characters (or "$", or "_") and
        may not start with a digit. Reserved keywords that may not be used as parameter
        names are "datum", "event", "item", and "parent".
    bind : dict, :class:`Binding`, :class:`BindInput`, :class:`BindRange`, :class:`BindDirect`, :class:`BindCheckbox`, :class:`BindRadioSelect`
        Binds the parameter to an external input element such as a slider, selection list or
        radio button group.
    expr : str, :class:`Expr`
        An expression for the value of the parameter. This expression may include other
        parameters, in which case the parameter will automatically update in response to
        upstream parameter changes.
    value : Any
        The `initial value <http://vega.github.io/vega-lite/docs/value.html>`__ of the
        parameter.

        **Default value:** ``undefined``
    """
    _schema = {'$ref': '#/definitions/VariableParameter'}

    def __init__(self, name: Union[str, 'SchemaBase', UndefinedType]=Undefined, bind: Union[dict, 'SchemaBase', UndefinedType]=Undefined, expr: Union[str, 'SchemaBase', UndefinedType]=Undefined, value: Union[Any, UndefinedType]=Undefined, **kwds):
        super(VariableParameter, self).__init__(name=name, bind=bind, expr=expr, value=value, **kwds)