import sys
import abc
import io
import copyreg
import pickle
from numba import cloudpickle
from llvmlite import ir
def _rebuild_reduction(cls, *args):
    """
    Global hook to rebuild a given class from its __reduce__ arguments.
    """
    return cls._rebuild(*args)