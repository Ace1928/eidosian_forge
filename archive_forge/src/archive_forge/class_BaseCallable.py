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
class BaseCallable(TraitType):
    """ A trait type whose value must be a Python callable.
    """
    metadata = {'copy': 'ref'}
    default_value_type = DefaultValue.constant
    default_value = None
    info_text = 'a callable value'

    def validate(self, object, name, value):
        """ Validates that the value is a Python callable.
        """
        if value is None or callable(value):
            return value
        self.error(object, name, value)