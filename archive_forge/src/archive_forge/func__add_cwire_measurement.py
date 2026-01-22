from dataclasses import dataclass
from typing import Optional
import pennylane as qml
from pennylane.measurements import (
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
def _add_cwire_measurement(m, layer_str, config):
    """Updates ``layer_str`` with the ``m`` measurement when it is used
    for collecting mid-circuit measurement statistics."""
    mcms = [v.measurements[0] for v in m.mv] if isinstance(m.mv, list) else m.mv.measurements
    layer_str = _add_cwire_measurement_grouping_symbols(mcms, layer_str, config)
    mv_label = 'MCM'
    meas_label = measurement_label_map[m.return_type](mv_label)
    n_wires = len(config.wire_map)
    for mcm in mcms:
        ind = config.bit_map[mcm] + n_wires
        layer_str[ind] += meas_label
    return layer_str