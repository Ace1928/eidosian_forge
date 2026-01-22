import abc
import os
import typing
import warnings
import weakref
import rpy2.rinterface
import rpy2.rinterface_lib.callbacks
from rpy2.robjects import conversion
def __getstate__(self):
    return (super().__getstate__(), self.__dict__.copy())