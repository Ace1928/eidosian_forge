from dataclasses import dataclass
from typing import Optional
import pennylane as qml
from pennylane.measurements import (
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
def _add_op(op, layer_str, config):
    """Updates ``layer_str`` with ``op`` operation."""
    if isinstance(op, qml.ops.Conditional):
        layer_str = _add_cond_grouping_symbols(op, layer_str, config)
        return _add_op(op.then_op, layer_str, config)
    if isinstance(op, MidMeasureMP):
        return _add_mid_measure_op(op, layer_str, config)
    layer_str = _add_grouping_symbols(op, layer_str, config)
    control_wires, control_values = unwrap_controls(op)
    if control_values:
        for w, val in zip(control_wires, control_values):
            layer_str[config.wire_map[w]] += '●' if val else '○'
    else:
        for w in control_wires:
            layer_str[config.wire_map[w]] += '●'
    label = op.label(decimals=config.decimals, cache=config.cache).replace('\n', '')
    if len(op.wires) == 0:
        for i, s in enumerate(layer_str):
            layer_str[i] = s + label
    else:
        for w in op.wires:
            if w not in control_wires:
                layer_str[config.wire_map[w]] += label
    return layer_str