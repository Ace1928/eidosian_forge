import logging
from io import StringIO
from operator import itemgetter, attrgetter
from pyomo.common.config import (
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.label import LPFileLabeler, NumericLabeler
from pyomo.opt import WriterFactory
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import (
from pyomo.core.base import Set, RangeSet, ExternalFunction
from pyomo.network import Port
def _normalize_constraint(data):
    (vid1, vid2), coef = data
    c1 = getVarOrder(vid1)
    c2 = getVarOrder(vid2)
    if c2 < c1:
        col = (c2, c1)
        sym = f' {getSymbol(getVar(vid2))} * {getSymbol(getVar(vid1))}\n'
    elif c1 == c2:
        col = (c1, c1)
        sym = f' {getSymbol(getVar(vid2))} ^ 2\n'
    else:
        col = (c1, c2)
        sym = f' {getSymbol(getVar(vid1))} * {getSymbol(getVar(vid2))}\n'
    if coef < 0:
        return (col, repr(coef) + sym)
    else:
        return (col, '+' + repr(coef) + sym)