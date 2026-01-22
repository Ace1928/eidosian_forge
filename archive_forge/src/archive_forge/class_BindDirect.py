from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class BindDirect(Binding):
    """BindDirect schema wrapper

    Parameters
    ----------

    element : str, dict, :class:`Element`
        An input element that exposes a *value* property and supports the `EventTarget
        <https://developer.mozilla.org/en-US/docs/Web/API/EventTarget>`__ interface, or a
        CSS selector string to such an element. When the element updates and dispatches an
        event, the *value* property will be used as the new, bound signal value. When the
        signal updates independent of the element, the *value* property will be set to the
        signal value and a new event will be dispatched on the element.
    debounce : float
        If defined, delays event handling until the specified milliseconds have elapsed
        since the last event was fired.
    event : str
        The event (default ``"input"`` ) to listen for to track changes on the external
        element.
    """
    _schema = {'$ref': '#/definitions/BindDirect'}

    def __init__(self, element: Union[str, dict, 'SchemaBase', UndefinedType]=Undefined, debounce: Union[float, UndefinedType]=Undefined, event: Union[str, UndefinedType]=Undefined, **kwds):
        super(BindDirect, self).__init__(element=element, debounce=debounce, event=event, **kwds)