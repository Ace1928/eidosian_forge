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
def trait_set(self, trait_change_notify=True, **traits):
    """ Shortcut for setting object trait attributes.

        Treats each keyword argument to the method as the name of a trait
        attribute and sets the corresponding trait attribute to the value
        specified. This is a useful shorthand when a number of trait attributes
        need to be set on an object, or a trait attribute value needs to be set
        in a lambda function. For example, you can write::

            person.trait_set(name='Bill', age=27)

        instead of::

            person.name = 'Bill'
            person.age = 27

        Parameters
        ----------
        trait_change_notify : bool
            If **True** (the default), then each value assigned may generate a
            trait change notification. If **False**, then no trait change
            notifications will be generated. (see also: trait_setq)
        **traits :
            Key/value pairs, the trait attributes and their values to be
            set

        Returns
        -------
        self :
            The method returns this object, after setting attributes.
        """
    if not trait_change_notify:
        self._trait_change_notify(False)
        try:
            for name, value in traits.items():
                setattr(self, name, value)
        finally:
            self._trait_change_notify(True)
    else:
        for name, value in traits.items():
            setattr(self, name, value)
    return self