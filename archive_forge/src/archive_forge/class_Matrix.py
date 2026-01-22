import abc
import collections.abc
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import sexp
from . import conversion
import rpy2.rlike.container as rlc
import datetime
import copy
import itertools
import math
import os
import jinja2  # type: ignore
import time
import tzlocal
from time import struct_time, mktime
import typing
import warnings
from rpy2.rinterface import (Sexp, ListSexpVector, StrSexpVector,
class Matrix(Array):
    """ An R matrix """
    _transpose = baseenv_ri['t']
    _rownames = baseenv_ri['rownames']
    _colnames = baseenv_ri['colnames']
    _dot = baseenv_ri['%*%']
    _matmul = baseenv_ri['%*%']
    _crossprod = baseenv_ri['crossprod']
    _tcrossprod = baseenv_ri['tcrossprod']
    _svd = baseenv_ri['svd']
    _eigen = baseenv_ri['eigen']

    def __nrow_get(self):
        """ Number of rows.
        :rtype: integer """
        return self.dim[0]
    nrow = property(__nrow_get, None, None, 'Number of rows')

    def __ncol_get(self):
        """ Number of columns.
        :rtype: integer """
        return self.dim[1]
    ncol = property(__ncol_get, None, None, 'Number of columns')

    def __rownames_get(self):
        """ Row names

        :rtype: SexpVector
        """
        res = self._rownames(self)
        return conversion.get_conversion().rpy2py(res)

    def __rownames_set(self, rn):
        if isinstance(rn, StrSexpVector):
            if len(rn) != self.nrow:
                raise ValueError('Invalid length.')
            if self.dimnames is NULL:
                dn = ListVector.from_length(2)
                dn[0] = rn
                self.do_slot_assign('dimnames', dn)
            else:
                dn = self.dimnames
                dn[0] = rn
        else:
            raise ValueError('The rownames attribute can only be an R string vector.')
    rownames = property(__rownames_get, __rownames_set, None, 'Row names')

    def __colnames_get(self):
        """ Column names

        :rtype: SexpVector
        """
        res = self._colnames(self)
        return conversion.get_conversion().rpy2py(res)

    def __colnames_set(self, cn):
        if isinstance(cn, StrSexpVector):
            if len(cn) != self.ncol:
                raise ValueError('Invalid length.')
            if self.dimnames is NULL:
                dn = ListVector.from_length(2)
                dn[1] = cn
                self.do_slot_assign('dimnames', dn)
            else:
                dn = self.dimnames
                dn[1] = cn
        else:
            raise ValueError('The colnames attribute can only be an R string vector.')
    colnames = property(__colnames_get, __colnames_set, None, 'Column names')

    def transpose(self):
        """ transpose the matrix """
        res = self._transpose(self)
        return conversion.get_conversion().rpy2py(res)

    def __matmul__(self, x):
        """ Matrix multiplication. """
        cv = conversion.get_conversion()
        res = self._matmul(self, cv.py2rpy(x))
        return cv.rpy2py(res)

    def crossprod(self, m):
        """ crossproduct X'.Y"""
        cv = conversion.get_conversion()
        res = self._crossprod(self, cv.rpy2py(m))
        return cv.rpy2py(res)

    def tcrossprod(self, m):
        """ crossproduct X.Y'"""
        res = self._tcrossprod(self, m)
        return conversion.get_conversion().rpy2py(res)

    def svd(self, nu=None, nv=None, linpack=False):
        """ SVD decomposition.
        If nu is None, it is given the default value min(tuple(self.dim)).
        If nv is None, it is given the default value min(tuple(self.dim)).
        """
        if nu is None:
            nu = min(tuple(self.dim))
        if nv is None:
            nv = min(tuple(self.dim))
        res = self._svd(self, nu=nu, nv=nv)
        return conversion.get_conversion().rpy2py(res)

    def dot(self, m):
        """ Matrix multiplication """
        res = self._dot(self, m)
        return conversion.get_conversion().rpy2py(res)

    def eigen(self):
        """ Eigen values """
        res = self._eigen(self)
        return conversion.get_conversion().rpy2py(res)