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
class SingletonHasTraits(HasTraits):
    """ Singleton class that support trait attributes.
    """

    @deprecated('SingletonHasTraits has been deprecated and will be removed in the future. Avoid using it')
    def __new__(cls, *args, **traits):
        if '_the_instance' not in cls.__dict__:
            cls._the_instance = HasTraits.__new__(cls, *args, **traits)
        return cls._the_instance