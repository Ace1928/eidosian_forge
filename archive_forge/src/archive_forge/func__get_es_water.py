import collections
import pytest
from ..util.testing import requires
from ..chemistry import Substance, Reaction, Equilibrium, Species
def _get_es_water(EqSys=None):
    if EqSys is None:
        EqSys = EqSystem
    H2O = Substance('H2O', charge=0, composition={1: 2, 8: 1})
    OHm = Substance('OH-', charge=-1, composition={1: 1, 8: 1})
    Hp = Substance('H+', charge=1, composition={1: 1})
    Kw = 1e-14 / 55.5
    w_auto_p = Equilibrium({'H2O': 1}, {'Hp': 1, 'OHm': 1}, Kw)
    return EqSys([w_auto_p], [H2O, OHm, Hp])