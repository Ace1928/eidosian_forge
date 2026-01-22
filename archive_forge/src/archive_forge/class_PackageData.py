import os
import typing
import warnings
from types import ModuleType
from warnings import warn
import rpy2.rinterface as rinterface
from . import conversion
from rpy2.robjects.functions import (SignatureTranslatedFunction,
from rpy2.robjects import Environment
from rpy2.robjects.packages_utils import (
import rpy2.robjects.help as rhelp
class PackageData(object):
    """ Datasets in an R package.
    In R datasets can be distributed with a package.

    Datasets can be:

    - serialized R objects

    - R code (that produces the dataset)

    For a given R packages, datasets are stored separately from the rest
    of the code and are evaluated/loaded lazily.

    The lazy aspect has been conserved and the dataset are only loaded
    or generated when called through the method 'fetch()'.
    """
    _packagename = None
    _lib_loc = None
    _datasets = None

    def __init__(self, packagename, lib_loc=rinterface.NULL):
        self._packagename = packagename
        self._lib_loc

    def _init_setlist(self):
        _datasets = dict()
        tmp_m = _data(**{'package': StrSexpVector((self._packagename,)), 'lib.loc': self._lib_loc})[2]
        nrows, ncols = tmp_m.do_slot('dim')
        c_i = 2
        for r_i in range(nrows):
            _datasets[tmp_m[r_i + c_i * nrows]] = None
        self._datasets = _datasets

    def names(self):
        """ Names of the datasets"""
        if self._datasets is None:
            self._init_setlist()
        return self._datasets.keys()

    def fetch(self, name):
        """ Fetch the dataset (loads it or evaluates the R associated
        with it.

        In R, datasets are loaded into the global environment by default
        but this function returns an environment that contains the dataset(s).
        """
        if self._datasets is None:
            self._init_setlist()
        if name not in self._datasets:
            raise KeyError('Data set "%s" cannot be found' % name)
        env = _new_env()
        _data(StrSexpVector((name,)), **{'package': StrSexpVector((self._packagename,)), 'lib.loc': self._lib_loc, 'envir': env})
        return Environment(env)