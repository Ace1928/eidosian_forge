import collections
import pytest
from ..util.testing import requires
from ..chemistry import Substance, Reaction, Equilibrium, Species
def _get_NaCl(Cls, **precipitate_kwargs):
    Na_p, Cl_m, NaCl = sbstncs = (Cls('Na+', 1, composition={11: 1}), Cls('Cl-', -1, composition={17: 1}), Cls('NaCl', composition={11: 1, 17: 1}, **precipitate_kwargs))
    sp = Equilibrium({'NaCl': 1}, {'Na+': 1, 'Cl-': 1}, 4.0)
    eqsys = EqSystem([sp], sbstncs)
    cases = [[(0, 0, 0.1), (0.1, 0.1, 0)], [(0.5, 0.5, 0.4), (0.9, 0.9, 0)], [(1, 1, 1), (2, 2, 0)], [(0, 0, 2), (2, 2, 0)], [(3, 3, 3), (2, 2, 4)], [(3, 3, 0), (2, 2, 1)], [(0, 0, 3), (2, 2, 1)]]
    return (eqsys, [s.name for s in sbstncs], cases)