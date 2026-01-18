from operator import itemgetter
from functorch.compile import make_boxed_func
import torch
import torch.nn as nn
from torch._functorch.compilers import aot_module
from torch._inductor.decomposition import select_decomp_table
from torch.distributed._tensor import DTensor

    Util to print the operator coverage summary of a certain model with tabulute.

    Must have tabulate module installed.
    