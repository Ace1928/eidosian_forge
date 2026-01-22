from collections import namedtuple
from functools import singledispatch
import pennylane as qml
from pennylane import ops
from pennylane.measurements import MidMeasureMP
from .mpldrawer import MPLDrawer
from .drawable_layers import drawable_layers
from .utils import convert_wire_order, unwrap_controls, cwire_connections, default_bit_map
from .style import _set_style
@singledispatch
def _add_operation_to_drawer(op: qml.operation.Operator, drawer: MPLDrawer, layer: int, config: _Config) -> None:
    """Adds the ``op`` to an ``MPLDrawer`` at the designated location.

    Args:
        op (.Operator): An operator to add to the drawer
        drawer (.MPLDrawer): A matplotlib drawer
        layer (int): The layer to place the operator in.
        config (_Config): named tuple containing ``wire_map``, ``decimals``, and ``active_wire_notches``.

    Side Effects:
        Adds a depiction of ``op`` to ``drawer``

    """
    op_control_wires, control_values = unwrap_controls(op)
    target_wires = [w for w in op.wires if w not in op_control_wires] if len(op.wires) != 0 else list(range(drawer.n_wires))
    if control_values is None:
        control_values = [True for _ in op_control_wires]
    if op_control_wires:
        drawer.ctrl(layer, op_control_wires, wires_target=target_wires, control_values=control_values)
    drawer.box_gate(layer, target_wires, op.label(decimals=config.decimals), box_options={'zorder': 4}, text_options={'zorder': 5}, active_wire_notches=config.active_wire_notches)