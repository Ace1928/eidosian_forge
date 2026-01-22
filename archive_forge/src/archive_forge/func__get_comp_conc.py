from collections import OrderedDict
from .. import Substance
def _get_comp_conc(rsys, odesys, comp_keys, skip_keys):
    comp_conc = []
    for comp_key in comp_keys:
        if comp_key in skip_keys:
            continue
        _d = OrderedDict()
        for si, subst_key in enumerate(odesys.names):
            coeff = rsys.substances[subst_key].composition.get(comp_key, 0)
            if coeff != 0:
                _d[si] = coeff
        comp_conc.append(_d)
    return comp_conc