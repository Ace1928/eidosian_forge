import pytest
import numpy as np
from ase.build import fcc111
from ase.ga.slab_operators import (CutSpliceSlabCrossover,
def test_random_permutation(seed, cu_slab):
    rng = np.random.RandomState(seed)
    p1 = cu_slab
    p1.symbols[:8] = 'Au'
    op = RandomSlabPermutation(rng=rng)
    child, desc = op.get_new_individual([p1])
    assert (child.symbols == 'Au').sum() == 8
    assert sum(p1.numbers == child.numbers) == 22