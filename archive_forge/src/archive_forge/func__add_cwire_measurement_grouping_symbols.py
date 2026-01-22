from dataclasses import dataclass
from typing import Optional
import pennylane as qml
from pennylane.measurements import (
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
def _add_cwire_measurement_grouping_symbols(mcms, layer_str, config):
    """Adds symbols indicating the extent of a given object for mid-circuit measurement
    statistics."""
    if len(mcms) > 1:
        n_wires = len(config.wire_map)
        mapped_bits = [config.bit_map[m] for m in mcms]
        min_b, max_b = (min(mapped_bits) + n_wires, max(mapped_bits) + n_wires)
        layer_str[min_b] = '╭'
        layer_str[max_b] = '╰'
        for b in range(min_b + 1, max_b):
            layer_str[b] = '├' if b - n_wires in mapped_bits else '│'
    return layer_str