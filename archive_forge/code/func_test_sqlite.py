import numpy as np
from ase import Atom
from ase.build import bulk
from ase.calculators.checkpoint import Checkpoint, CheckpointCalculator
from ase.calculators.lj import LennardJones
from ase.lattice.cubic import Diamond
def test_sqlite(testdir):
    CP = Checkpoint('checkpoints.db')
    a = Diamond('Si', size=[2, 2, 2])
    a = CP(op1)(a, 1.0)
    op1a = a.copy()
    a, ra = CP(op2)(a, 2.0)
    op2a = a.copy()
    op2ra = ra.copy()
    CP = Checkpoint('checkpoints.db')
    a = Diamond('Si', size=[2, 2, 2])
    a = CP(op1)(a, 1.0)
    assert a == op1a
    a, ra = CP(op2)(a, 2.0)
    assert a == op2a
    assert np.abs(ra - op2ra).max() < 1e-05