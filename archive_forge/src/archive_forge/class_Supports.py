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
class Supports(Instance):
    """ A trait type whose value is adapted to a specified protocol.

    In other words, the value of the trait directly provide, or can be adapted
    to, the given protocol (Interface or type).

    The value of the trait after assignment is the possibly adapted value
    (i.e., it is the original assigned value if that provides the protocol,
    or is an adapter otherwise).

    The original, unadapted value is stored in a "shadow" attribute with
    the same name followed by an underscore (e.g., ``foo`` and ``foo_``).
    """
    adapt_default = 'yes'

    def post_setattr(self, object, name, value):
        """ Performs additional post-assignment processing.
        """
        object.__dict__[name + '_'] = value

    def as_ctrait(self):
        """ Returns a CTrait corresponding to the trait defined by this class.
        """
        return self.modify_ctrait(super().as_ctrait())

    def modify_ctrait(self, ctrait):
        ctrait.post_setattr_original_value = True
        return ctrait