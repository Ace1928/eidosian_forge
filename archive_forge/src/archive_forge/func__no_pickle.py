import sys
import abc
import io
import copyreg
import pickle
from numba import cloudpickle
from llvmlite import ir
def _no_pickle(obj):
    raise pickle.PicklingError(f'Pickling of {type(obj)} is unsupported')