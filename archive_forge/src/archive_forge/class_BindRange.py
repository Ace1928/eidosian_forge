from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class BindRange(Binding):
    """BindRange schema wrapper

    Parameters
    ----------

    input : str

    debounce : float
        If defined, delays event handling until the specified milliseconds have elapsed
        since the last event was fired.
    element : str, :class:`Element`
        An optional CSS selector string indicating the parent element to which the input
        element should be added. By default, all input elements are added within the parent
        container of the Vega view.
    max : float
        Sets the maximum slider value. Defaults to the larger of the signal value and
        ``100``.
    min : float
        Sets the minimum slider value. Defaults to the smaller of the signal value and
        ``0``.
    name : str
        By default, the signal name is used to label input elements. This ``name`` property
        can be used instead to specify a custom label for the bound signal.
    step : float
        Sets the minimum slider increment. If undefined, the step size will be automatically
        determined based on the ``min`` and ``max`` values.
    """
    _schema = {'$ref': '#/definitions/BindRange'}

    def __init__(self, input: Union[str, UndefinedType]=Undefined, debounce: Union[float, UndefinedType]=Undefined, element: Union[str, 'SchemaBase', UndefinedType]=Undefined, max: Union[float, UndefinedType]=Undefined, min: Union[float, UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, step: Union[float, UndefinedType]=Undefined, **kwds):
        super(BindRange, self).__init__(input=input, debounce=debounce, element=element, max=max, min=min, name=name, step=step, **kwds)