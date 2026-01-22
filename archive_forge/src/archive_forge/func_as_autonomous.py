from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def as_autonomous(self, new_indep_name=None, new_latex_indep_name=None):
    if self.autonomous_exprs:
        return self
    old_indep_name = self.indep_name or _get_indep_name(self.names)
    new_names = () if not self.names else self.names + (old_indep_name,)
    new_indep_name = new_indep_name or _get_indep_name(new_names)
    new_latex_indep_name = new_latex_indep_name or new_indep_name
    new_latex_names = () if not self.latex_names else self.latex_names + (new_latex_indep_name,)
    new_indep = self.be.Symbol(new_indep_name)
    new_dep = self.dep + (self.indep,)
    new_exprs = self.exprs + (self.indep ** 0,)
    new_kw = dict(names=new_names, indep_name=new_indep_name, latex_names=new_latex_names, latex_indep_name=new_latex_indep_name, autonomous_interface=False)
    if new_names:
        new_kw['taken_names'] = self.taken_names + (old_indep_name,)
    if self.linear_invariants:
        new_kw['linear_invariants'] = np.concatenate((self.linear_invariants, np.zeros((self.linear_invariants.shape[0], 1))), axis=-1)
    for attr in filter(lambda k: k not in new_kw, self._attrs_to_copy):
        new_kw[attr] = getattr(self, attr)

    def autonomous_post_processor(x, y, p):
        try:
            y[0][0, 0]
        except:
            pass
        else:
            return zip(*[autonomous_post_processor(_x, _y, _p) for _x, _y, _p in zip(x, y, p)])
        return (y[..., -1], y[..., :-1], p)
    new_kw['post_processors'] = [autonomous_post_processor] + self.post_processors
    new_kw['_indep_autonomous_key'] = new_names[-1] if new_names else True
    return self.__class__(zip(new_dep, new_exprs), indep=new_indep, params=self.params, **new_kw)