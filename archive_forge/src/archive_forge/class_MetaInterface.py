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
class MetaInterface(ABCMetaHasTraits):
    """ Meta class for interfaces.

    Interfaces are simple ABCs with the following features:-

    1) They cannot be instantiated (they are interfaces, not implementations!).
    2) Calling them is equivalent to calling 'adapt'.

    """

    @deprecated('use "adapt(adaptee, protocol)" instead.')
    def __call__(self, adaptee, default=AdaptationError):
        """ Attempt to adapt the adaptee to this interface.

        Note that this means that (intentionally ;^) that interfaces
        cannot be instantiated!

        """
        from traits.adaptation.api import adapt
        return adapt(adaptee, self, default=default)