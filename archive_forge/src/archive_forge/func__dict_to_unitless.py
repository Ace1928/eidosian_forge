from collections import OrderedDict, defaultdict
from itertools import chain
from chempy.kinetics.ode import get_odesys
from chempy.units import to_unitless, linspace, logspace_from_lin
def _dict_to_unitless(d, u):
    return {k: to_unitless(v, u) for k, v in d.items()}