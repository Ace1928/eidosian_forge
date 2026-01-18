import logging
import operator
from typing import Callable, List, Optional, Set, Tuple
from functorch import make_fx
import torch
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table

    Lower Inductor compatible portions of the graph module to Inductor.

    Args:
        node_predicate: user predicate for determining whether to consider a node for
            lowering.
        subgraph_predicate: user predicate for determining whether to consider a list of
            candidate nodes for lowering.
        dumper: a callback for dumping subgraphs for human digestion. For exmaple, it
            can be a function that writes to disk/blob storage and returns the
            path/handle. The returned path/handle for each subgraph will be made
            available in the subgraph call node in the parent graph, as well as the
            label of the profiler block for the subgraph.
    