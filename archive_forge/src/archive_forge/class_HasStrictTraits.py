import abc
import copy as copy_module
import inspect
import os
import pickle
import re
import types
import warnings
import weakref
from types import FunctionType
from . import __version__ as TraitsVersion
from .adaptation.adaptation_error import AdaptationError
from .constants import DefaultValue, TraitKind
from .ctrait import CTrait, __newobj__
from .ctraits import CHasTraits
from .observation import api as observe_api
from .traits import (
from .trait_types import Any, Bool, Disallow, Event, Python, Str
from .trait_notifiers import (
from .trait_base import (
from .trait_errors import TraitError
from .util.deprecated import deprecated
from .util._traitsui_helpers import check_traitsui_major_version
from .trait_converters import check_trait, mapped_trait_for, trait_for
class HasStrictTraits(HasTraits):
    """ This class guarantees that any object attribute that does not have an
    explicit or wildcard trait definition results in an exception.

    This feature can be useful in cases where a more rigorous software
    engineering approach is being used than is typical for Python programs. It
    also helps prevent typos and spelling mistakes in attribute names from
    going unnoticed; a misspelled attribute name typically causes an exception.
    """
    _ = Disallow