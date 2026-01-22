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
class BaseBytes(TraitType):
    """ A trait type whose value must be a bytestring.
    """
    default_value_type = DefaultValue.constant
    default_value = b''
    info_text = 'a bytes string'
    encoding = None

    def validate(self, object, name, value):
        """ Validates that a specified value is valid for this trait.

        Note: The 'fast validator' version performs this check in C.
        """
        if isinstance(value, bytes):
            return value
        self.error(object, name, value)

    def create_editor(self):
        """ Returns the default traits UI editor for this type of trait.
        """
        from .traits import bytes_editor
        auto_set = self.auto_set
        if auto_set is None:
            auto_set = True
        enter_set = self.enter_set or False
        return bytes_editor(auto_set, enter_set, self.encoding)