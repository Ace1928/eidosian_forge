import logging
from pyomo.core.base.units_container import units, UnitsError
from pyomo.core.base import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.network import Port, Arc
from pyomo.mpec import Complementarity
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.numvalue import native_types
from pyomo.util.components import iter_component
from pyomo.common.collections import ComponentSet
def _assert_units_consistent_arc_data(arcdata):
    """
    Raise an exception if the any units do not match for the connected ports
    """
    sport = arcdata.source
    dport = arcdata.destination
    if sport is None or dport is None:
        return
    for key in sport.vars:
        svar = sport.vars[key]
        dvar = dport.vars[key]
        if svar.is_indexed():
            for k in svar:
                svardata = svar[k]
                dvardata = dvar[k]
                assert_units_equivalent(svardata, dvardata)
        else:
            assert_units_equivalent(svar, dvar)