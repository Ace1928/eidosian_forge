import collections
import pytest
from ..util.testing import requires
from ..chemistry import Substance, Reaction, Equilibrium, Species
def _species(Cls):
    return (Cls('H2O', 0, composition={1: 2, 8: 1}), Cls('H+', 1, composition={1: 1}), Cls('OH-', -1, composition={1: 1, 8: 1}), Cls('NH4+', 1, composition={1: 4, 7: 1}), Cls('NH3', 0, composition={1: 3, 7: 1}))