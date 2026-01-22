from importlib import import_module
import sys
from types import FunctionType, MethodType
from .constants import DefaultValue, ValidateTrait
from .trait_base import (
from .trait_base import RangeTypes  # noqa: F401, used by TraitsUI
from .trait_errors import TraitError
from .trait_dict_object import TraitDictEvent, TraitDictObject
from .trait_converters import trait_from
from .trait_handler import TraitHandler
from .trait_list_object import TraitListEvent, TraitListObject
from .util.deprecated import deprecated
class TraitEnum(TraitHandler):
    """ Ensures that a value assigned to a trait attribute is a member of a
    specified list of values.

    TraitEnum is the underlying handler for the forms of the Trait() function
    that take a list of possible values

    The list of legal values can be provided as a list or tuple of values.
    That is, ``TraitEnum([1, 2, 3])``, ``TraitEnum((1, 2, 3))`` and
    ``TraitEnum(1, 2, 3)`` are equivalent. For example::

        class Flower(HasTraits):
            color = Trait('white', TraitEnum(['white', 'yellow', 'red']))
            kind  = Trait('annual', TraitEnum('annual', 'perennial'))

    This example defines a Flower class, which has a **color** trait
    attribute, which can have as its value, one of the three strings,
    'white', 'yellow', or 'red', and a **kind** trait attribute, which can
    have as its value, either of the strings 'annual' or 'perennial'. This
    is equivalent to the following class definition::

        class Flower(HasTraits):
            color = Trait(['white', 'yellow', 'red'])
            kind  = Trait('annual', 'perennial')

    The Trait() function automatically maps traits of the form shown in
    this example to the form shown in the preceding example whenever it
    encounters them in a trait definition.

    Parameters
    ----------
    *values
        Either all legal values for the enumeration, or a single list or tuple
        of the legal values.

    Attributes
    ----------
    values : tuple
        Enumeration of all legal values for a trait.
    """

    def __init__(self, *values):
        if len(values) == 1 and type(values[0]) in SequenceTypes:
            values = values[0]
        self.values = tuple(values)
        self.fast_validate = (ValidateTrait.enum, self.values)

    def validate(self, object, name, value):
        if value in self.values:
            return value
        self.error(object, name, value)

    def info(self):
        return ' or '.join([repr(x) for x in self.values])

    def get_editor(self, trait):
        from traitsui.api import EnumEditor
        return EnumEditor(values=self, cols=trait.cols or 3, evaluate=trait.evaluate, mode=trait.mode or 'radio')