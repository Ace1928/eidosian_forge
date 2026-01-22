import copy
import os
import pickle
import warnings
import numpy as np
class sliceGenerator(object):
    """Just a compact way to generate tuples of slice objects."""

    def __getitem__(self, arg):
        return arg

    def __getslice__(self, arg):
        return arg