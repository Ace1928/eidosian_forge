import logging
import sys
from copy import deepcopy
from pickle import PickleError
from weakref import ref as weakref_ref
import pyomo.common
from pyomo.common import DeveloperError
from pyomo.common.autoslots import AutoSlots, fast_deepcopy
from pyomo.common.collections import OrderedDict
from pyomo.common.deprecation import (
from pyomo.common.factory import Factory
from pyomo.common.formatting import tabular_writer, StreamIndenter
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base.component_namer import name_repr, index_repr
from pyomo.core.base.global_set import UnindexedComponent_index
def _pop_from_kwargs(self, name, kwargs, namelist, notset=None):
    args = [arg for arg in (kwargs.pop(name, notset) for name in namelist) if arg is not notset]
    if len(args) == 1:
        return args[0]
    elif not args:
        return notset
    else:
        argnames = "%s%s '%s='" % (', '.join(("'%s='" % _ for _ in namelist[:-1])), ',' if len(namelist) > 2 else '', namelist[-1])
        raise ValueError('Duplicate initialization: %s() only accepts one of %s' % (name, argnames))