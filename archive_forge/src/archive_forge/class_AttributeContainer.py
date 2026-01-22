from collections import defaultdict, namedtuple, OrderedDict
from functools import wraps
from itertools import product
import os
import types
import warnings
from .. import __url__
from .deprecation import Deprecation
class AttributeContainer(object):
    """Used to turn e.g. a dictionary to a module-like object.

    Parameters
    ----------
    \\*\\*kwargs : dictionary

    Examples
    --------
    >>> def RT(T, const):
    ...     return T*const.molar_gas_constant
    ...
    >>> from quantities import constants
    >>> RT(273.15, constants)
    array(273.15) * R
    >>> my_constants = AttributeContainer(molar_gas_constant=42)
    >>> RT(273.15, my_constants)
    11472.3

    """

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def as_dict(self):
        return self.__dict__.copy()

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(set(dir(self)) - set(dir(object()))))