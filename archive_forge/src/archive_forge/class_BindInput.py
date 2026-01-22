from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class BindInput(Binding):
    """BindInput schema wrapper

    Parameters
    ----------

    autocomplete : str
        A hint for form autofill. See the `HTML autocomplete attribute
        <https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/autocomplete>`__ for
        additional information.
    debounce : float
        If defined, delays event handling until the specified milliseconds have elapsed
        since the last event was fired.
    element : str, :class:`Element`
        An optional CSS selector string indicating the parent element to which the input
        element should be added. By default, all input elements are added within the parent
        container of the Vega view.
    input : str
        The type of input element to use. The valid values are ``"checkbox"``, ``"radio"``,
        ``"range"``, ``"select"``, and any other legal `HTML form input type
        <https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input>`__.
    name : str
        By default, the signal name is used to label input elements. This ``name`` property
        can be used instead to specify a custom label for the bound signal.
    placeholder : str
        Text that appears in the form control when it has no value set.
    """
    _schema = {'$ref': '#/definitions/BindInput'}

    def __init__(self, autocomplete: Union[str, UndefinedType]=Undefined, debounce: Union[float, UndefinedType]=Undefined, element: Union[str, 'SchemaBase', UndefinedType]=Undefined, input: Union[str, UndefinedType]=Undefined, name: Union[str, UndefinedType]=Undefined, placeholder: Union[str, UndefinedType]=Undefined, **kwds):
        super(BindInput, self).__init__(autocomplete=autocomplete, debounce=debounce, element=element, input=input, name=name, placeholder=placeholder, **kwds)