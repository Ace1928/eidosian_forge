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
class TraitCoerceType(TraitHandler):
    """Ensures that a value assigned to a trait attribute is of a specified
    Python type, or can be coerced to the specified type.

    TraitCoerceType is the underlying handler for the predefined traits and
    factories for Python simple types. The TraitCoerceType class is also an
    example of a parametrized type, because the single TraitCoerceType class
    allows creating instances that check for totally different sets of values.
    For example::

        class Person(HasTraits):
            name = Trait('', TraitCoerceType(''))
            weight = Trait(0.0, TraitCoerceType(float))

    In this example, the **name** attribute must be of type ``str`` (string),
    while the **weight** attribute must be of type ``float``, although both are
    based on instances of the TraitCoerceType class. Note that this example is
    essentially the same as writing::

        class Person(HasTraits):
            name = Trait('')
            weight = Trait(0.0)

    This simpler form is automatically changed by the Trait() function into
    the first form, based on TraitCoerceType instances, when the trait
    attributes are defined.

    For attributes based on TraitCoerceType instances, if a value that is
    assigned is not of the type defined for the trait, a TraitError exception
    is raised. However, in certain cases, if the value can be coerced to the
    required type, then the coerced value is assigned to the attribute. Only
    *widening* coercions are allowed, to avoid any possible loss of precision.
    The following table lists the allowed coercions.

    ============ =================
     Trait Type   Coercible Types
    ============ =================
    complex      float, int
    float        int
    ============ =================

    Parameters
    ----------
    aType : type or object
        Either a Python type or a Python value.  If this is an object, it is
        mapped to its corresponding type. For example, the string 'cat' is
        automatically mapped to ``str``.

    Attributes
    ----------
    aType : type
        A Python type to coerce values to.
   """

    def __init__(self, aType):
        if not isinstance(aType, type):
            aType = type(aType)
        self.aType = aType
        try:
            self.fast_validate = CoercableTypes[aType]
        except:
            self.fast_validate = (ValidateTrait.coerce, aType)

    def validate(self, object, name, value):
        fv = self.fast_validate
        tv = type(value)
        if tv is fv[1]:
            return value
        for typei in fv[2:]:
            if tv is typei:
                return fv[1](value)
        self.error(object, name, value)

    def info(self):
        return 'a value of %s' % str(self.aType)[1:-1]

    def get_editor(self, trait):
        if self.aType is bool:
            if self.editor is None:
                from traitsui.api import BooleanEditor
                self.editor = BooleanEditor()
            return self.editor
        auto_set = trait.auto_set
        if auto_set is None:
            auto_set = True
        from traitsui.api import TextEditor
        return TextEditor(auto_set=auto_set, enter_set=trait.enter_set or False, evaluate=self.fast_validate[1])