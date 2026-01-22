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
def _add_event_handlers(trait, cls, handlers):
    """ Adds any specified event handlers defined for a trait by a class.
    """
    events = trait.event
    if events is not None:
        if isinstance(events, str):
            events = [events]
        for event in events:
            handlers.append(_get_method(cls, '_%s_changed' % event))
            handlers.append(_get_method(cls, '_%s_fired' % event))