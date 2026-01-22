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
class TraitFunction(TraitHandler):
    """Ensures that assigned trait attribute values are acceptable to a
    specified validator function.

    TraitFunction is the underlying handler for the predefined trait
    **Function**, and for the use of function references as arguments to the
    Trait() function.

    The signature of the function must be of the form *function*(*object*,
    *name*, *value*). The function must verify that *value* is a legal value
    for the *name* trait attribute of *object*.  If it is, the value returned
    by the function is the actual value assigned to the trait attribute. If it
    is not, the function must raise a TraitError exception.

    Parameters
    ----------
    aFunc : function
        A function to validate trait attribute values.

    Attributes
    ----------
    aFunc : function
        A function to validate trait attribute values.
    """

    def __init__(self, aFunc):
        if not isinstance(aFunc, CallableTypes):
            raise TraitError('Argument must be callable.')
        self.aFunc = aFunc
        self.fast_validate = (ValidateTrait.function, aFunc)

    def validate(self, object, name, value):
        try:
            return self.aFunc(object, name, value)
        except TraitError:
            self.error(object, name, value)

    def info(self):
        try:
            return self.aFunc.info
        except:
            if self.aFunc.__doc__:
                return self.aFunc.__doc__
            return 'a legal value'