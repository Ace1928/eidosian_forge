from dataclasses import dataclass
from typing import Optional
import pennylane as qml
from pennylane.measurements import (
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
def _add_cond_grouping_symbols(op, layer_str, config):
    """Adds symbols indicating the extent of a given object for conditional
    operators"""
    n_wires = len(config.wire_map)
    mapped_wires = [config.wire_map[w] for w in op.wires]
    mapped_bits = [config.bit_map[m] for m in op.meas_val.measurements]
    max_w = max(mapped_wires)
    max_b = max(mapped_bits) + n_wires
    ctrl_symbol = '╩' if config.cur_layer != config.cwire_layers[max(mapped_bits)][-1] else '╝'
    layer_str[max_b] = f'═{ctrl_symbol}'
    for w in range(max_w + 1, max(config.wire_map.values()) + 1):
        layer_str[w] = '─║'
    for b in range(n_wires, max_b):
        if b - n_wires in mapped_bits:
            intersection = '╣' if config.cur_layer == config.cwire_layers[b - n_wires][-1] else '╬'
            layer_str[b] = f'═{intersection}'
        else:
            filler = ' ' if layer_str[b][-1] == ' ' else '═'
            layer_str[b] = f'{filler}║'
    return layer_str