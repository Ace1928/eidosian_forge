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