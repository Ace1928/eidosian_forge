import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import (
def test_shared_blocks(self):
    v = MPIBlockVector(3, [0, 1, -1], comm)
    rank = comm.Get_rank()
    if rank == 0:
        v.set_block(0, np.arange(3))
    if rank == 1:
        v.set_block(1, np.arange(4))
    v.set_block(2, np.arange(2))
    shared = v.shared_blocks
    self.assertTrue(np.allclose(np.array([2]), shared))