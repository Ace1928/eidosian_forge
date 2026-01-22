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
class TraitCastType(TraitCoerceType):
    """Ensures that a value assigned to a trait attribute is of a specified
    Python type, or can be cast to the specified type.

    This class is similar to TraitCoerceType, but uses casting rather than
    coercion. Values are cast by calling the type with the value to be assigned
    as an argument. When casting is performed, the result of the cast is the
    value assigned to the trait attribute.

    Any trait that uses a TraitCastType instance in its definition ensures that
    its value is of the type associated with the TraitCastType instance. For
    example::

        class Person(HasTraits):
            name = Trait('', TraitCastType(''))
            weight = Trait(0.0, TraitCastType(float))

    In this example, the **name** trait must be of type ``str`` (string), while
    the **weight** trait must be of type ``float``. Note that this example is
    essentially the same as writing::

        class Person(HasTraits):
            name = CStr
            weight = CFloat

    To understand the difference between TraitCoerceType and TraitCastType (and
    also between Float and CFloat), consider the following example::

        >>> class Person(HasTraits):
        ...     weight = Float
        ...     cweight = CFloat
        ...
        >>> bill = Person()
        >>> bill.weight = 180    # OK, coerced to 180.0
        >>> bill.cweight = 180   # OK, cast to 180.0
        >>> bill.weight = '180'  # Error, invalid coercion
        >>> bill.cweight = '180' # OK, cast to float('180')

    Parameters
    ----------
    aType : type
        Either a Python type or a Python value.  If this is an object, it is
        mapped to its corresponding type. For example, the string 'cat' is
        automatically mapped to ``str``.

    Attributes
    ----------
    aType : type
        A Python type to cast values to.
    """

    def __init__(self, aType):
        if not isinstance(aType, type):
            aType = type(aType)
        self.aType = aType
        self.fast_validate = (ValidateTrait.cast, aType)

    def validate(self, object, name, value):
        if type(value) is self.aType:
            return value
        try:
            return self.aType(value)
        except:
            self.error(object, name, value)