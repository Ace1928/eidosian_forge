import math
import ctypes
import copy
from .random import uniform
from .symbol import Symbol
from . import symbol
from ..base import _LIB, check_call
from ..base import SymbolHandle, _as_list
from ..attribute import AttrScope
def _create_subgraph(graph_vars, graph_func, subgraph_name):
    subgraph_name = _get_unique_subgraph_name(subgraph_name)
    with AttrScope(__subgraph_name__=subgraph_name):
        new_graph_vars = [symbol.var(sym.name) for sym in graph_vars]
        outputs = graph_func(*new_graph_vars)
        outputs, out_fmt = _flatten(outputs, 'cond outputs')
        num_outputs = len(outputs)
        all_input_names = symbol.Group(outputs).list_inputs()
        in_input = lambda x: x.name in all_input_names
        in_graph = lambda x: x.list_attr().get('__subgraph_name__', '') == subgraph_name
        make_identity = lambda x: symbol.op.identity(x) if in_input(x) or not in_graph(x) else x
        graph = symbol.Group(list(map(make_identity, outputs)))
    return (graph, num_outputs, out_fmt)