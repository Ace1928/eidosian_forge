import numpy as np
from pyodesys.util import import_
from pyodesys.core import integrate_chained
from pyodesys.symbolic import SymbolicSys, PartiallySolvedSystem, symmetricsys, TransformedSys
from pyodesys.tests._robertson import get_ode_exprs
class TransformedNativeSys(TransformedSys, NativeSys):
    pass