from collections import namedtuple
import time
import logging
import warnings
import sys
import numpy as np
from scipy import spatial
from scipy.optimize import OptimizeResult, minimize, Bounds
from scipy.optimize._optimize import MemoizeJac
from scipy.optimize._constraints import new_bounds_to_old
from scipy.optimize._minimize import standardize_constraints
from scipy._lib._util import _FunctionWrapper
from scipy.optimize._shgo_lib._complex import Complex
class LMapCache:

    def __init__(self):
        self.cache = {}
        self.v_maps = []
        self.xl_maps = []
        self.xl_maps_set = set()
        self.f_maps = []
        self.lbound_maps = []
        self.size = 0

    def __getitem__(self, v):
        try:
            v = np.ndarray.tolist(v)
        except TypeError:
            pass
        v = tuple(v)
        try:
            return self.cache[v]
        except KeyError:
            xval = LMap(v)
            self.cache[v] = xval
            return self.cache[v]

    def add_res(self, v, lres, bounds=None):
        v = np.ndarray.tolist(v)
        v = tuple(v)
        self.cache[v].x_l = lres.x
        self.cache[v].lres = lres
        self.cache[v].f_min = lres.fun
        self.cache[v].lbounds = bounds
        self.size += 1
        self.v_maps.append(v)
        self.xl_maps.append(lres.x)
        self.xl_maps_set.add(tuple(lres.x))
        self.f_maps.append(lres.fun)
        self.lbound_maps.append(bounds)

    def sort_cache_result(self):
        """
        Sort results and build the global return object
        """
        results = {}
        self.xl_maps = np.array(self.xl_maps)
        self.f_maps = np.array(self.f_maps)
        ind_sorted = np.argsort(self.f_maps)
        results['xl'] = self.xl_maps[ind_sorted]
        self.f_maps = np.array(self.f_maps)
        results['funl'] = self.f_maps[ind_sorted]
        results['funl'] = results['funl'].T
        results['x'] = self.xl_maps[ind_sorted[0]]
        results['fun'] = self.f_maps[ind_sorted[0]]
        self.xl_maps = np.ndarray.tolist(self.xl_maps)
        self.f_maps = np.ndarray.tolist(self.f_maps)
        return results