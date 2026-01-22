from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class BindRadioSelect(Binding):
    """BindRadioSelect schema wrapper

    Parameters
    ----------

    input : Literal['radio', 'select']

    options : Sequence[Any]
        An array of options to select from.
    debounce : float
        If defined, delays event handling until the specified milliseconds have elapsed
        since the last event was fired.
    element : str, :class:`Element`
        An optional CSS selector string indicating the parent element to which the input
        element should be added. By default, all input elements are added within the parent
        container of the Vega view.
    labels : Sequence[str]
        An array of label strings to represent the ``options`` values. If unspecified, the
        ``options`` value will be coerced to a string and used as the label.
    name : str
        By default, the signal name is used to label input elements. This ``name`` property
        can be used instead to specify a custom label for the bound signal.
    """
    _schema = {'$ref': '#/definitions/BindRadioSelect'}

    def __init__(self, input: Union[Literal['radio', 'select'], UndefinedType]=Undefined, options: Union[Sequence[Any], UndefinedType]=Undefined, debounce: Union[float, UndefinedType]=Undefined, element: Union[str, 'SchemaBase', UndefinedType]=Undefined, labels: Union[Sequence[str], UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, **kwds):
        super(BindRadioSelect, self).__init__(input=input, options=options, debounce=debounce, element=element, labels=labels, name=name, **kwds)