import logging
import operator
from typing import Callable, List, Optional, Set, Tuple
from functorch import make_fx
import torch
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
class _InductorModule(torch.nn.Module):

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        super().__init__()
        self.gm = gm
        self.compiled: Optional[Callable[[List[torch.Tensor]], List[torch.Tensor]]] = None

    def forward(self, *args: torch.Tensor, tag: str) -> List[torch.Tensor]:
        if self.compiled is None:
            inductor_decompositions = select_decomp_table()
            decomp_gm = make_fx(self.gm, decomposition_table=inductor_decompositions)(*args)
            logger.info('Lowering subgraph (%s) to Inductor...', tag)
            self.compiled = compile_fx_inner(decomp_gm, list(args), cudagraphs=False)
            logger.info('Completed lowering subgraph (%s) to Inductor', tag)
        with torch.profiler.record_function(tag):
            assert self.compiled is not None
            return self.compiled(list(args))