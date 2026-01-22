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
class HasRequiredTraits(HasStrictTraits):
    """ This class builds on the functionality of HasStrictTraits and ensures
    that any object attribute with `required=True` in its metadata must be
    passed as an argument on object initialization.

    This can be useful in cases where an object has traits which are required
    for it to function correctly.

    Raises
    ------
    TraitError
        If a required trait is not passed as an argument.

    Examples
    --------
    A class with required traits:

    >>> class RequiredTest(HasRequiredTraits):
    ...     required_trait = Any(required=True)
    ...     non_required_trait = Any()

    Creating an instance of a HasRequiredTraits subclass:

    >>> test_instance = RequiredTest(required_trait=13, non_required_trait=11)
    >>> test_instance2 = RequiredTest(required_trait=13)

    Forgetting to specify a required trait:

    >>> test_instance = RequiredTest(non_required_trait=11)
    traits.trait_errors.TraitError: The following required traits were not
    provided: required_trait.
    """

    def __init__(self, **traits):
        missing_required_traits = [name for name in self.trait_names(required=True) if name not in traits]
        if missing_required_traits:
            raise TraitError('The following required traits were not provided: {}.'.format(', '.join(sorted(missing_required_traits))))
        super().__init__(**traits)