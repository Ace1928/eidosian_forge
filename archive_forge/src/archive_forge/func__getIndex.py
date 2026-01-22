import copy
import os
import pickle
import warnings
import numpy as np
def _getIndex(self, axis, name):
    ax = self._info[axis]
    if ax is not None and 'cols' in ax:
        for i in range(0, len(ax['cols'])):
            if 'name' in ax['cols'][i] and ax['cols'][i]['name'] == name:
                return i
    raise Exception('Axis %d has no column named %s.\n  info=%s' % (axis, name, self._info))