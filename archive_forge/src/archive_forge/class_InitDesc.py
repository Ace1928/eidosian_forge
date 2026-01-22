import re
import logging
import warnings
import json
from math import sqrt
import numpy as np
from .base import string_types
from .ndarray import NDArray, load
from . import random
from . import registry
from . import ndarray
from . util import is_np_array
from . import numpy as _mx_np  # pylint: disable=reimported
class InitDesc(str):
    """
    Descriptor for the initialization pattern.

    Parameters
    ----------
    name : str
        Name of variable.
    attrs : dict of str to str
        Attributes of this variable taken from ``Symbol.attr_dict``.
    global_init : Initializer
        Global initializer to fallback to.
    """

    def __new__(cls, name, attrs=None, global_init=None):
        ret = super(InitDesc, cls).__new__(cls, name)
        ret.attrs = attrs or {}
        ret.global_init = global_init
        return ret