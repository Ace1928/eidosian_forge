import importlib
import numpy
from . import object_arrays
from . import cupy as _cupy
from . import jax as _jax
from . import tensorflow as _tensorflow
from . import theano as _theano
from . import torch as _torch
def _import_func(func, backend, default=None):
    """Try and import ``{backend}.{func}``.
    If library is installed and func is found, return the func;
    otherwise if default is provided, return default;
    otherwise raise an error.
    """
    try:
        lib = importlib.import_module(_aliases.get(backend, backend))
        return getattr(lib, func) if default is None else getattr(lib, func, default)
    except AttributeError:
        error_msg = "{} doesn't seem to provide the function {} - see https://optimized-einsum.readthedocs.io/en/latest/backends.html for details on which functions are required for which contractions."
        raise AttributeError(error_msg.format(backend, func))