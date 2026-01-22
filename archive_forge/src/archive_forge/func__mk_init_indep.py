from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def _mk_init_indep(self, name, be=None, prefix='i_', suffix=''):
    name = name or 'indep'
    be = be or self.be
    name = prefix + str(name) + suffix
    if getattr(self, 'indep', None) is not None:
        if self.indep.name == name:
            raise ValueError('Name ambiguity in independent variable name')
    return self._Symbol(name, be)