import copy
import os
import pickle
import warnings
import numpy as np
def _axisCopy(self, i):
    return copy.deepcopy(self._info[i])