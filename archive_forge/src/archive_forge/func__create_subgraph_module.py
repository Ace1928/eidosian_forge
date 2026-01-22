import logging
import operator
from typing import Callable, List, Optional, Set, Tuple
from functorch import make_fx
import torch
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
def _create_subgraph_module(inputs: List[torch.fx.Node], body: List[torch.fx.Node], outputs: List[torch.fx.Node]) -> torch.fx.GraphModule:
    subgraph: torch.fx.Graph = torch.fx.Graph()
    node_to_subgraph_node = {}
    for idx, inp in enumerate(inputs):
        subgraph_inp = subgraph.placeholder(name=f'arg_{idx}')
        subgraph_inp.meta = inp.meta
        node_to_subgraph_node[inp] = subgraph_inp
    for node in body:
        subgraph_node = subgraph.node_copy(node, arg_transform=lambda x: node_to_subgraph_node[x])
        node_to_subgraph_node[node] = subgraph_node
    subgraph.output(result=tuple((node_to_subgraph_node[x] for x in outputs)))
    subgraph.eliminate_dead_code()
    subgraph.lint()
    return torch.fx.GraphModule(root={}, graph=subgraph)