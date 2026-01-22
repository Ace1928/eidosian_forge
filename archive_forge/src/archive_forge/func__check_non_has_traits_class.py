from inspect import getfullargspec, getmro
import logging
from types import FunctionType
from .has_traits import HasTraits
def _check_non_has_traits_class(self, cls, interface, error_mode):
    """ Checks that a non-'HasTraits' class implements an interface.
        """
    return self._check_methods(cls, interface, error_mode)