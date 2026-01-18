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
def wrapped_class(klass):
    for protocol in protocols:
        type(protocol).register(protocol, klass)
    if CHECK_INTERFACES:
        from .interface_checker import check_implements
        warnings.warn('In the future, the @provides decorator will not perform interface checks. Set has_traits.CHECK_INTERFACES to 0 to suppress this warning.', DeprecationWarning, stacklevel=2)
        check_implements(klass, protocols, CHECK_INTERFACES)
    return klass