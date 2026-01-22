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
class Disallow(TraitType):
    """ A trait that prevents any value from being assigned or read.

    Any attempt to get or set the value of the trait attribute raises an
    exception. This trait is most often used in conjunction with wildcard
    naming, for example, to catch spelling mistakes in attribute names.

    See the *Traits User Manual* for details on wildcards.
    """
    ctrait_type = TraitKind.disallow
    default_value_type = DefaultValue.constant