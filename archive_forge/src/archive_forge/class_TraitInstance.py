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
class TraitInstance(TraitHandler):
    """Ensures that trait attribute values belong to a specified Python type.

    Any trait that uses a TraitInstance handler ensures that its values belong
    to the specified type or class (or one of its subclasses). For example::

        class Employee(HasTraits):
            manager = Trait(None, TraitInstance(Employee, True))

    This example defines a class Employee, which has a **manager** trait
    attribute, which accepts either None or an instance of Employee
    as its value.

    TraitInstance ensures that assigned values are exactly of the type
    specified (i.e., no coercion is performed).

    Parameters
    ----------
    aClass : type, object or str
        A Python type or a string that identifies the type, or an object.
        If this is an object, it is mapped to the class it is an instance of.
        If this is a str, it is either the name  of a class in the module
        identified by the module parameter, or an identifier of the form
        "*module_name*[.*module_name*....].*class_name*".
    allow_none : bool
        Flag indicating whether None is accepted as a valid value.
    module : str
        The name of the module that the class belongs to.  This is ignored if
        the type is provided directly, or the str value is an identifier with
        '.'s in it.

    Attributes
    ----------
    aClass : type or str
        A Python type, or a string which identifies the type.  If this is a
        str, it is either the name of a class in the module identified by the
        module attribute, or an identifier of the form
        "*module_name*[.*module_name*....].*class_name*".  A string value will
        be replaced by the actual type object the first time the trait is used
        to validate an object.
    module : str
        The name of the module that the class belongs to.  This is ignored if
        the type is provided directly, or the str value is an identifier with
        '.'s in it.
    """

    def __init__(self, aClass, allow_none=True, module=''):
        self._allow_none = allow_none
        self.module = module
        if isinstance(aClass, str):
            self.aClass = aClass
        else:
            if not isinstance(aClass, type):
                aClass = aClass.__class__
            self.aClass = aClass
            self.set_fast_validate()

    def allow_none(self):
        """ Whether or not None is permitted as a valid value.

        Returns
        -------
        bool
            Whether or not None is a valid value.
        """
        self._allow_none = True
        if hasattr(self, 'fast_validate'):
            self.set_fast_validate()

    def set_fast_validate(self):
        fast_validate = [ValidateTrait.instance, self.aClass]
        if self._allow_none:
            fast_validate = [ValidateTrait.instance, None, self.aClass]
        if self.aClass in TypeTypes:
            fast_validate[0] = ValidateTrait.type
        self.fast_validate = tuple(fast_validate)

    def validate(self, object, name, value):
        if value is None:
            if self._allow_none:
                return value
            else:
                self.error(object, name, value)
        if isinstance(self.aClass, str):
            self.resolve_class(object, name, value)
        if isinstance(value, self.aClass):
            return value
        self.error(object, name, value)

    def info(self):
        aClass = self.aClass
        if type(aClass) is not str:
            aClass = aClass.__name__
        result = class_of(aClass)
        if self._allow_none:
            return result + ' or None'
        return result

    def resolve_class(self, object, name, value):
        aClass = self.validate_class(self.find_class(self.aClass))
        if aClass is None:
            self.error(object, name, value)
        self.aClass = aClass
        self.set_fast_validate()
        trait = object.base_trait(name)
        handler = trait.handler
        if handler is not self and hasattr(handler, 'item_trait'):
            trait = handler.item_trait
        trait.set_validate(self.fast_validate)

    def find_class(self, klass):
        module = self.module
        col = klass.rfind('.')
        if col >= 0:
            module = klass[:col]
            klass = klass[col + 1:]
        theClass = getattr(sys.modules.get(module), klass, None)
        if theClass is None and col >= 0:
            try:
                mod = import_module(module)
                theClass = getattr(mod, klass, None)
            except Exception:
                pass
        return theClass

    def validate_class(self, aClass):
        return aClass

    def create_default_value(self, *args, **kw):
        aClass = args[0]
        if isinstance(aClass, str):
            aClass = self.validate_class(self.find_class(aClass))
            if aClass is None:
                raise TraitError('Unable to locate class: ' + args[0])
        return aClass(*args[1:], **kw)

    def get_editor(self, trait):
        if self.editor is None:
            from traitsui.api import InstanceEditor
            self.editor = InstanceEditor(label=trait.label or '', view=trait.view or '', kind=trait.kind or 'live')
        return self.editor