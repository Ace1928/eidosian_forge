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
def trait_views(self, klass=None):
    """ Returns a list of the names of all view elements associated with
        the current object's class.

        If *klass* is specified, the list of names is filtered such that only
        objects that are instances of the specified class are returned.

        Parameters
        ----------
        klass : class
            A class, such that all returned names must correspond to instances
            of this class. Possible values include:

            * Group
            * Item
            * View
            * ViewElement
            * ViewSubElement
        """
    view_elements = self.__class__.__dict__[ViewTraits]
    if isinstance(view_elements, dict):
        view_elements = self._init_trait_view_elements()
    return view_elements.filter_by(klass)