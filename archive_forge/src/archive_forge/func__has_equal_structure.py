import operator
from ..dependencies import numpy as np
from .base_block import BaseBlockVector
def _has_equal_structure(self, other):
    """
        Parameters
        ----------
        other: BlockVector

        Returns
        -------
        equal_structure: bool
            True if self and other have the same block structure (recursive). False otherwise.
        """
    if not isinstance(other, BlockVector):
        return False
    if self.nblocks != other.nblocks:
        return False
    for ndx, block1 in enumerate(self):
        block2 = other.get_block(ndx)
        if isinstance(block1, BlockVector):
            if not isinstance(block2, BlockVector):
                return False
            if not block1._has_equal_structure(block2):
                return False
        elif isinstance(block2, BlockVector):
            return False
    return True