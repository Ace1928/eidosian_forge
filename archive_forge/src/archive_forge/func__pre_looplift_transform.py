from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def _pre_looplift_transform(func_ir):
    """Canonicalize loops for looplifting.
    """
    from numba.core.postproc import PostProcessor
    cfg = compute_cfg_from_blocks(func_ir.blocks)
    for loop_info in cfg.loops().values():
        if _has_multiple_loop_exits(cfg, loop_info):
            func_ir, _common_key = _fix_multi_exit_blocks(func_ir, loop_info.exits)
    func_ir._reset_analysis_variables()
    PostProcessor(func_ir).run()
    return func_ir