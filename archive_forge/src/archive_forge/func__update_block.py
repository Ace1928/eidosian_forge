import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core import Suffix, Var, Constraint, Piecewise, Block
from pyomo.core import Expression, Param
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.block import IndexedBlock, SortComponents
from pyomo.dae import ContinuousSet, DAE_Error
from pyomo.common.formatting import tostr
from io import StringIO
def _update_block(blk):
    """
    This method will construct any additional indices in a block
    resulting from the discretization of a ContinuousSet. For
    Block-derived components we check if the Block construct method has
    been overridden. If not then we update it like a regular block. If
    construct has been overridden then we try to call the component's
    update_after_discretization method. If the component hasn't
    implemented this method then we throw a warning and try to update it
    like a normal block. The issue, when construct is overridden, is that
    anything could be happening and we can't automatically assume that
    treating the block-derived component like a normal block will be
    sufficient to update it correctly.

    """
    if blk.construct.__func__ is not getattr(IndexedBlock.construct, '__func__', IndexedBlock.construct):
        if hasattr(blk, 'update_after_discretization'):
            blk.update_after_discretization()
            return
        else:
            logger.warning('DAE(misc): Attempting to apply a discretization transformation to the Block-derived component "%s". The component overrides the Block construct method but no update_after_discretization() function was found. Will attempt to update as a standard Block but user should verify that the component was expanded correctly. To suppress this warning, please provide an update_after_discretization() function on Block-derived components that override construct()' % blk.name)
    missing_idx = getattr(blk, '_dae_missing_idx', set([]))
    for idx in list(missing_idx):
        blk[idx]
    if hasattr(blk, '_dae_missing_idx'):
        del blk._dae_missing_idx