from pennylane.ops import Controlled, Conditional
from pennylane.measurements import MeasurementProcess, MidMeasureMP, MeasurementValue
def unwrap_controls(op):
    """Unwraps nested controlled operations for drawing.

    Controlled operations may themselves contain controlled operations; check
    for any nesting of operators when drawing so that we correctly identify
    and label _all_ control and target qubits.

    Args:
        op (.Operation): A PennyLane operation.

    Returns:
        Wires, List: The control wires of the operation, along with any associated
        control values.
    """
    control_wires = getattr(op, 'control_wires', [])
    control_values = getattr(op, 'hyperparameters', {}).get('control_values', None)
    if isinstance(control_values, list):
        control_values = control_values.copy()
    if isinstance(op, Controlled):
        next_ctrl = op
        while hasattr(next_ctrl, 'base'):
            if isinstance(next_ctrl.base, Controlled):
                base_control_wires = getattr(next_ctrl.base, 'control_wires', [])
                control_wires += base_control_wires
                base_control_values = next_ctrl.base.hyperparameters.get('control_values', [True] * len(base_control_wires))
                if control_values is not None:
                    control_values.extend(base_control_values)
            next_ctrl = next_ctrl.base
    control_values = [bool(int(i)) for i in control_values] if control_values else control_values
    return (control_wires, control_values)