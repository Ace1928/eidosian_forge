import operator
from ..dependencies import numpy as np
from .base_block import BaseBlockVector
def toMPIBlockVector(self, rank_ownership, mpi_comm, assert_correct_owners=False):
    """
        Creates a parallel MPIBlockVector from this BlockVector

        Parameters
        ----------
        rank_ownership: array_like
            Array_like of size nblocks. Each entry defines ownership of each block.
            There are two types of ownership. Block that are owned by all processor,
            and blocks owned by a single processor. If a block is owned by all
            processors then its ownership is -1. Otherwise, if a block is owned by
            a single processor, then its ownership is equal to the rank of the
            processor.
        mpi_comm: MPI communicator
            An MPI communicator. Tyically MPI.COMM_WORLD

        """
    from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
    assert_block_structure(self)
    assert len(rank_ownership) == self.nblocks, 'rank_ownership must be of size {}'.format(self.nblocks)
    mpi_bv = MPIBlockVector(self.nblocks, rank_ownership, mpi_comm, assert_correct_owners=assert_correct_owners)
    for bid in mpi_bv.owned_blocks:
        mpi_bv.set_block(bid, self.get_block(bid))
    return mpi_bv