from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict as ptDict, Type as ptType
import itertools
import weakref
from functools import cached_property
import numpy as np
from numba.core.utils import get_hashable_key
def _type_reconstructor(reconstructor, reconstructor_args, state):
    """
    Rebuild function for unpickling types.
    """
    obj = reconstructor(*reconstructor_args)
    if state:
        obj.__dict__.update(state)
    return type(obj)._intern(obj)