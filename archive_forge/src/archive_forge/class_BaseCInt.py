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
class BaseCInt(BaseInt):
    """ A coercing trait type whose value is an integer.
    """
    evaluate = int

    def validate(self, object, name, value):
        """ Validates that a specified value is valid for this trait.

        Note: The 'fast validator' version performs this check in C.
        """
        try:
            return int(value)
        except (ValueError, TypeError):
            self.error(object, name, value)