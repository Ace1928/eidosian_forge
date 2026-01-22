import os
import atexit
import functools
import pickle
import sys
import time
import warnings
import numpy as np
class ParallelModuleWrapper:

    def __getattr__(self, name):
        if name == 'rank' or name == 'size':
            warnings.warn('ase.parallel.{name} has been deprecated.  Please use ase.parallel.world.{name} instead.'.format(name=name), FutureWarning)
            return getattr(world, name)
        return getattr(_parallel, name)