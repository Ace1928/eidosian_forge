from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class BindCheckbox(Binding):
    """BindCheckbox schema wrapper

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
    name : str
        By default, the signal name is used to label input elements. This ``name`` property
        can be used instead to specify a custom label for the bound signal.
    """
    _schema = {'$ref': '#/definitions/BindCheckbox'}

    def __init__(self, input: Union[str, UndefinedType]=Undefined, debounce: Union[float, UndefinedType]=Undefined, element: Union[str, 'SchemaBase', UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, **kwds):
        super(BindCheckbox, self).__init__(input=input, debounce=debounce, element=element, name=name, **kwds)