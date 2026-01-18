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
def trait_view(self, name=None, view_element=None):
    """ Gets or sets a ViewElement associated with an object's class.

        If both *name* and *view_element* are specified, the view element is
        associated with *name* for the current object's class. (That is,
        *view_element* is added to the ViewElements object associated with
        the current object's class, indexed by *name*.)

        If only *name* is specified, the function returns the view element
        object associated with *name*, or None if *name* has no associated
        view element. View elements retrieved by this function are those that
        are bound to a class attribute in the class definition, or that are
        associated with a name by a previous call to this method.

        If neither *name* nor *view_element* is specified, the method returns a
        View object, based on the following order of preference:

        1. If there is a View object named ``traits_view`` associated with the
           current object, it is returned.
        2. If there is exactly one View object associated the current
           object, it is returned.
        3. Otherwise, it returns a View object containing items for all the
           non-event trait attributes on the current object.

        Parameters
        ----------
        name : str
            Name of a view element
        view_element : ViewElement
            View element to associate

        Returns
        -------
        A view element.
        """
    return self.__class__._trait_view(name, view_element, self.default_traits_view, self.trait_view_elements, self.visible_traits, self)