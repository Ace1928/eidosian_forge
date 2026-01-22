from dataclasses import dataclass
from typing import Optional
import pennylane as qml
from pennylane.measurements import (
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
def _add_measurement(m, layer_str, config):
    """Updates ``layer_str`` with the ``m`` measurement."""
    if m.mv is not None:
        return _add_cwire_measurement(m, layer_str, config)
    layer_str = _add_grouping_symbols(m, layer_str, config)
    if m.obs is None:
        obs_label = None
    else:
        obs_label = m.obs.label(decimals=config.decimals, cache=config.cache).replace('\n', '')
    if m.return_type in measurement_label_map:
        meas_label = measurement_label_map[m.return_type](obs_label)
    else:
        meas_label = m.return_type.value
    if len(m.wires) == 0:
        for i, s in enumerate(layer_str):
            layer_str[i] = s + meas_label
    for w in m.wires:
        layer_str[config.wire_map[w]] += meas_label
    return layer_str