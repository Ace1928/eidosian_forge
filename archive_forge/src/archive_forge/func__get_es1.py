import collections
import pytest
from ..util.testing import requires
from ..chemistry import Substance, Reaction, Equilibrium, Species
def _get_es1():
    a, b = sbstncs = (Species('a'), Species('b'))
    rxns = [Equilibrium({'a': 1}, {'b': 1}, 3)]
    return EqSystem(rxns, sbstncs)