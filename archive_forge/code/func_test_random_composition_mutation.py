import pytest
import numpy as np
from ase.build import fcc111
from ase.ga.slab_operators import (CutSpliceSlabCrossover,
def test_random_composition_mutation(seed, cu_slab):
    rng = np.random.RandomState(seed)
    p1 = cu_slab
    p1.symbols[3] = 'Au'
    op = RandomCompositionMutation(element_pools=['Cu', 'Au'], allowed_compositions=[(12, 12), (18, 6)], rng=rng)
    child, _ = op.get_new_individual([p1])
    no_Au = (child.symbols == 'Au').sum()
    assert no_Au in [6, 12]
    op = RandomCompositionMutation(element_pools=['Cu', 'Au'], rng=rng)
    child2 = op.operate(child)
    assert (child2.symbols == 'Au').sum() != no_Au