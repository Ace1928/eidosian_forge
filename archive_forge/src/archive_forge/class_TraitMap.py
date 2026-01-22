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
class TraitMap(TraitHandler):
    """ Checks that the value assigned to a trait attribute is a key of a
    specified dictionary, and also assigns the dictionary value corresponding
    to that key to a *shadow* attribute.

    A trait attribute that uses a TraitMap handler is called *mapped* trait
    attribute. In practice, this means that the resulting object actually
    contains two attributes: one whose value is a key of the TraitMap
    dictionary, and the other whose value is the corresponding value of the
    TraitMap dictionary. The name of the shadow attribute is simply the base
    attribute name with an underscore ('_') appended. Mapped trait attributes
    can be used to allow a variety of user-friendly input values to be mapped
    to a set of internal, program-friendly values.

    Example
    -------

    The following example defines a ``Person`` class::

        >>> class Person(HasTraits):
        ...     married = Trait('yes', TraitMap({'yes': 1, 'no': 0 })
        ...
        >>> bob = Person()
        >>> print bob.married
        yes
        >>> print bob.married_
        1

    In this example, the default value of the ``married`` attribute of the
    Person class is 'yes'. Because this attribute is defined using
    TraitPrefixList, instances of Person have another attribute,
    ``married_``, whose default value is 1, the dictionary value corresponding
    to the key 'yes'.

    Parameters
    ----------
    map : dict
        A dictionary whose keys are valid values for the trait attribute,
        and whose corresponding values are the values for the shadow
        trait attribute.

    Attributes
    ----------
    map : dict
        A dictionary whose keys are valid values for the trait attribute,
        and whose corresponding values are the values for the shadow
        trait attribute.
    """
    is_mapped = True

    def __init__(self, map):
        self.map = map
        self.fast_validate = (ValidateTrait.map, map)

    def validate(self, object, name, value):
        try:
            if value in self.map:
                return value
        except:
            pass
        self.error(object, name, value)

    def mapped_value(self, value):
        """ Get the mapped value for a value. """
        return self.map[value]

    def post_setattr(self, object, name, value):
        try:
            setattr(object, name + '_', self.mapped_value(value))
        except:
            raise TraitError('Unmappable')

    def info(self):
        keys = sorted((repr(x) for x in self.map.keys()))
        return ' or '.join(keys)

    def get_editor(self, trait):
        from traitsui.api import EnumEditor
        return EnumEditor(values=self, cols=trait.cols or 3)