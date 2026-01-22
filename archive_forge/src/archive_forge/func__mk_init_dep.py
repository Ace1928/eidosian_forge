from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def _mk_init_dep(self, names=None, be=None, ny=None, prefix='i_', suffix=''):
    be = be or self.be
    ny = ny or self.ny
    names = names or getattr(self, 'names', [str(i) for i in range(ny)])
    if getattr(self, 'dep', None) is not None:
        for dep in self.dep:
            if dep.name.startswith(prefix):
                raise ValueError('Name ambiguity in dependent variable names')
    use_names = names is not None and len(names) > 0
    return tuple((self._Symbol(prefix + names[idx] if use_names else str(idx) + suffix, be) for idx in range(ny)))