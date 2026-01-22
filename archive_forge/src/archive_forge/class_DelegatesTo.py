import collections.abc
import datetime
from importlib import import_module
import operator
from os import fspath
from os.path import isfile, isdir
import re
import sys
from types import FunctionType, MethodType, ModuleType
import uuid
import warnings
from .constants import DefaultValue, TraitKind, ValidateTrait
from .trait_base import (
from .trait_converters import trait_from, trait_cast
from .trait_dict_object import TraitDictEvent, TraitDictObject
from .trait_errors import TraitError
from .trait_list_object import TraitListEvent, TraitListObject
from .trait_set_object import TraitSetEvent, TraitSetObject
from .trait_type import (
from .traits import (
from .util.deprecated import deprecated
from .util.import_symbol import import_symbol
from .editor_factories import (
class DelegatesTo(Delegate):
    """ A trait type that matches the 'delegate' design pattern.

    This defines a trait whose value and definition is "delegated" to
    another trait on a different object.

    An object containing a delegator trait attribute must contain a
    second attribute that references the object containing the delegate
    trait attribute. The name of this second attribute is passed as the
    *delegate* argument to the DelegatesTo() function.

    The following rules govern the application of the prefix parameter:

    * If *prefix* is empty or omitted, the delegation is to an attribute
      of the delegate object with the same name as the delegator
      attribute.
    * If *prefix* is a valid Python attribute name, then the delegation
      is to an attribute whose name is the value of *prefix*.
    * If *prefix* ends with an asterisk ('*') and is longer than one
      character, then the delegation is to an attribute whose name is
      the value of *prefix*, minus the trailing asterisk, prepended to
      the delegator attribute name.
    * If *prefix* is equal to a single asterisk, the delegation is to an
      attribute whose name is the value of the delegator object's
      __prefix__ attribute prepended to delegator attribute name.

    Note that any changes to the delegator attribute are actually
    applied to the corresponding attribute on the delegate object. The
    original object containing the delegator trait is not modified.

    Parameters
    ----------
    delegate : str
        Name of the attribute on the current object which references
        the object that is the trait's delegate.
    prefix : str
        A prefix or substitution applied to the original attribute when
        looking up the delegated attribute.
    listenable : bool
        Indicates whether a listener can be attached to this attribute
        such that changes to the delegated attribute will trigger it.
    **metadata
        Trait metadata for the trait.
    """

    def __init__(self, delegate, prefix='', listenable=True, **metadata):
        super().__init__(delegate, prefix=prefix, modify=True, listenable=listenable, **metadata)