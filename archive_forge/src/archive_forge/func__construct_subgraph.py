import math
import ctypes
import copy
from .random import uniform
from .symbol import Symbol
from . import symbol
from ..base import _LIB, check_call
from ..base import SymbolHandle, _as_list
from ..attribute import AttrScope
def _construct_subgraph(sym_out, sym_states, name):
    sym_out = _as_list(sym_out)
    sym_states = _as_list(sym_states)
    all_outputs = []
    all_outputs.extend(sym_out)
    all_outputs.extend(sym_states)
    g = symbol.Group(all_outputs)
    flat_out = []
    all_input_names = g.list_inputs()
    output_names = {o.name for o in sym_out}
    for o in sym_out:
        if o.name in all_input_names or o.list_attr().get('__subgraph_name__', '') != name:
            flat_out.append(symbol.op.identity(o))
        else:
            flat_out.append(o)
    for s in sym_states:
        if s.name in all_input_names or s.name in output_names or s.list_attr().get('__subgraph_name__', '') != name:
            flat_out.append(symbol.op.identity(s))
        else:
            flat_out.append(s)
    return symbol.Group(flat_out)