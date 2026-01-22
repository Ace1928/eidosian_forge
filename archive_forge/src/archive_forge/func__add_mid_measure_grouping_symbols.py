from dataclasses import dataclass
from typing import Optional
import pennylane as qml
from pennylane.measurements import (
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
def _add_mid_measure_grouping_symbols(op, layer_str, config):
    """Adds symbols indicating the extent of a given object for mid-measure
    operators"""
    if op not in config.bit_map:
        return layer_str
    n_wires = len(config.wire_map)
    mapped_wire = config.wire_map[op.wires[0]]
    bit = config.bit_map[op] + n_wires
    layer_str[bit] += ' ╚'
    for w in range(mapped_wire + 1, n_wires):
        layer_str[w] += '─║'
    for b in range(n_wires, bit):
        filler = ' ' if layer_str[b][-1] == ' ' else '═'
        layer_str[b] += f'{filler}║'
    return layer_str