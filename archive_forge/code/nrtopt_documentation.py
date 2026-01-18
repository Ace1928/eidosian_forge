import re
from collections import defaultdict, deque
from llvmlite import binding as ll
from numba.core import cgutils

    Remove redundant reference count operations from the
    `llvmlite.binding.ModuleRef`. This parses the ll_module as a string and
    line by line to remove the unnecessary nrt refct pairs within each block.
    Decref calls are moved after the last incref call in the block to avoid
    temporarily decref'ing to zero (which can happen due to hidden decref from
    alias).

    Note: non-threadsafe due to usage of global LLVMcontext
    