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
class CList(List):
    """ A coercing trait type for a list of values of the specified type.
    """

    def validate(self, object, name, value):
        """ Validates that the values is a valid list.
        """
        if not isinstance(value, list):
            try:
                value = list(value)
            except (ValueError, TypeError):
                value = [value]
        return super().validate(object, name, value)

    def full_info(self, object, name, value):
        """ Returns a description of the trait.
        """
        return '%s or %s' % (self.item_trait.full_info(object, name, value), super().full_info(object, name, value))