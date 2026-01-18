from pyomo.core import Var
from pyomo.core.base.indexed_component import UnindexedComponent_set
def tighten_var_domain(comp, new_var, index_set=None):
    if index_set is None:
        if comp.is_indexed():
            index_set = comp.index_set()
        else:
            index_set = UnindexedComponent_set
    if comp.is_indexed():
        for i in index_set:
            try:
                _tighten(comp[i], new_var[i])
            except AttributeError:
                break
    else:
        try:
            _tighten(comp, new_var)
        except AttributeError:
            pass
    return new_var